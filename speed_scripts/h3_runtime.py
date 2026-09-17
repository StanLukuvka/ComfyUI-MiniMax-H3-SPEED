"""MiniMax-H3 SPEED multi-stage runtime.

Each resolution stage is one ``guider.sample()`` call. Between calls the
runtime converts the H3 video/audio state, expands video frequencies, aligns
the boundary sigma, resizes I2V keyframes, and re-enters at the next grid.
"""

from __future__ import annotations

import logging
import math

import torch

from .config import SpeedConfig
from .flow import (
    aligned_sigma,
    carry_preserved_audio,
    clock_reindex_audio_state,
    reentry_noise,
    time_shift_sigma,
    to_internal_state,
)
from .latent_class import LatentWalker
from .sampler_support import (
    SamplerCapability,
    SpeedSamplerHandle,
    SpeedTransition,
    create_speed_sampler_handle,
)
from .spectral import (
    dct2,
    dct_temporal,
    idct2,
    idct_temporal,
    lowpass_dct,
    spectral_expand,
    spectral_expand_3d,
)

log = logging.getLogger(__name__)

# Kept as a compatibility marker for tests/older callers. The walker is now
# purely local to one run and is no longer attached to the guider.
_LW_ATTR = "_speed_latent_walker"


class _OverrideSamplerHandle(SpeedSamplerHandle):
    """Test seam for injecting a sampler object without ComfyUI lookup."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.capability = SamplerCapability.STATELESS_STEP_LOCAL


def stage_resolution(config: SpeedConfig, stage_idx: int, full_h: int, full_w: int, full_t: int):
    """Return ``(h, w, t)`` for one configured stage."""
    scale = config.scales[stage_idx]
    height = max(1, round(full_h * scale))
    width = max(1, round(full_w * scale))
    if config.temporal_scales:
        frames = max(1, round(full_t * config.temporal_scales[stage_idx]))
    else:
        frames = full_t
    return height, width, frames


def power_at_frequency(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum ``P(omega) = A * |omega|**(-beta)``."""
    return A * abs(omega) ** (-beta)


def activation_threshold(power: float, delta: float) -> float:
    """Return the SPEED activation threshold for one radial frequency."""
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")
    return 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))


def unpack_latent(samples):
    """Validate and unpack the H3 nested latent into ``(video, audio)``."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax-H3 SPEED requires a NestedTensor video/audio latent")
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("MiniMax-H3 SPEED requires exactly video and audio streams")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("expected H3 video [B,C,T,H,W] and audio [B,C,2,T]")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax-H3 supports batch size one")
    if audio.shape[2] != 2:
        raise ValueError("MiniMax-H3 audio latent requires stereo axis size two")
    return video, audio


def pack_latent(video, audio):
    """Pack video/audio streams into ComfyUI's nested latent container."""
    from comfy import nested_tensor

    return nested_tensor.NestedTensor([video, audio])


def resolve_sigma_shifts(guider):
    """Return ``(video_shift, audio_shift, audio_scale)`` for the active H3 model."""
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None)
    if model is None:
        raise ValueError("no model_patcher.model on guider")

    candidates = []
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        model_options = getattr(patcher, "model_options", None)
    if isinstance(model_options, dict):
        transformer_options = model_options.get("transformer_options", {})
        if isinstance(transformer_options, dict):
            candidates.append((
                transformer_options.get("minimax_h3_sigma_shift_video"),
                transformer_options.get("minimax_h3_sigma_shift_audio"),
            ))

    candidates.append((
        getattr(model, "sigma_shift_video", None),
        getattr(model, "sigma_shift_audio", None),
    ))
    diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is not None:
        candidates.append((
            getattr(diffusion_model, "sigma_shift_video", None),
            getattr(diffusion_model, "sigma_shift_audio", None),
        ))

    shifts = next(
        (pair for pair in candidates if all(isinstance(value, (int, float)) for value in pair)),
        None,
    )
    if shifts is None:
        raise ValueError(
            "active MiniMax-H3 sigma shifts are unavailable: expected "
            "sigma_shift_video/audio on the H3 model or transformer options"
        )
    video_shift, audio_shift = map(float, shifts)
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError("active MiniMax-H3 shifts must be positive")
    return video_shift, audio_shift, video_shift / audio_shift


def _build_preview_callback(guider, total_steps, x0_output):
    """Build ComfyUI's normal latent-preview callback when available."""
    try:
        import latent_preview
    except Exception as exc:
        log.info("[SPEED-preview] latent_preview unavailable (%r) — previews disabled", exc)
        return None
    return latent_preview.prepare_callback(guider.model_patcher, total_steps, x0_output)


def _wrap_preview_callback(stock_cb, capture_state, global_offset, global_total):
    """Map one stage-local callback onto the run-wide progress timeline."""
    if stock_cb is None:
        try:
            import comfy.utils as comfy_utils

            pbar = comfy_utils.ProgressBar(global_total)
        except Exception:
            pbar = None

        def callback(step, x0, x, total_steps):
            capture_state["x0"] = x0
            if pbar is not None:
                try:
                    pbar.update_absolute(global_offset + step + 1, global_total)
                except Exception:
                    pass

        return callback

    warned = False

    def callback(step, x0, x, total_steps):
        nonlocal warned
        capture_state["x0"] = x0
        try:
            stock_cb(global_offset + step, x0, x, global_total)
        except Exception as exc:
            if not warned:
                warned = True
                log.warning(
                    "[SPEED-preview] preview callback failed (%r) — disabling updates for this run",
                    exc,
                )

    return callback


def _find_first_step_below(sigmas, threshold: float) -> int:
    values = [float(sigma) for sigma in sigmas]
    last = len(values) - 1
    for index in range(last):
        if values[index] <= threshold:
            return index
    return last


def resolve_transition_steps(
    config: SpeedConfig,
    sigmas,
    H_full: int | None = None,
    W_full: int | None = None,
) -> tuple[int, ...]:
    """Resolve global sigma indices for every resolution transition."""
    if config.transition_mode == "explicit":
        return config.transition_steps

    if H_full is None or W_full is None:
        H_full, W_full = config.full_latent_h, config.full_latent_w

    omega_max = min(H_full, W_full) / 2.0
    steps = []
    for scale in config.scales[:-1]:
        power = power_at_frequency(
            scale * omega_max,
            config.noise_amplitude,
            config.noise_decay_exponent,
        )
        threshold = activation_threshold(power, config.delta)
        steps.append(_find_first_step_below(sigmas, threshold))
    return tuple(steps)


def _coupled_transition(
    internal_video,
    full_noise_video,
    target_thw,
    sigma: float,
):
    """Expand with one fixed full-grid spectral noise field."""
    target_t, target_h, target_w = target_thw
    source_t, source_h, source_w = internal_video.shape[-3:]
    full_noise_video = full_noise_video.to(
        device=internal_video.device,
        dtype=internal_video.dtype,
    )
    full_coefficients = dct2(dct_temporal(full_noise_video))
    target_coefficients = (
        full_coefficients[..., :target_t, :target_h, :target_w].clone() * float(sigma)
    )
    target_coefficients[..., :source_t, :source_h, :source_w] = dct2(
        dct_temporal(internal_video)
    )
    return idct_temporal(idct2(target_coefficients)).to(dtype=internal_video.dtype)


def _expand_video(
    internal_video,
    *,
    target_thw,
    sigma: float,
    seed: int,
    full_noise_video=None,
):
    """Expand the carried video state to the next stage geometry."""
    target_t, target_h, target_w = target_thw
    if full_noise_video is not None:
        return _coupled_transition(
            internal_video,
            full_noise_video,
            target_thw,
            sigma,
        )
    if target_t > internal_video.shape[-3]:
        return spectral_expand_3d(
            internal_video,
            target_thw,
            sigma,
            seed,
        )
    return spectral_expand(
        internal_video,
        (target_h, target_w),
        sigma,
        seed,
    )


def run_speed_pipeline(
    noise,
    guider,
    sigmas: torch.Tensor,
    latent: dict,
    config: SpeedConfig,
    *,
    sampler_name: str = "euler",
    sampler_override=None,
    disable_pbar: bool = False,
    output_device=None,
    preview_callback=None,
    x0_output=None,
):
    """Run the configured progressive-resolution SPEED chain."""
    if "noise_mask" in latent:
        raise ValueError("T2V oracle does not support noise masks")

    full_video, full_audio = unpack_latent(latent.get("samples"))
    if torch.count_nonzero(full_video) or torch.count_nonzero(full_audio):
        raise ValueError("T2V oracle currently requires an empty H3 latent")
    if sigmas.ndim != 1 or len(sigmas) < 3:
        raise ValueError("sigmas must be a one-dimensional schedule")

    video_shift, audio_shift, audio_scale = resolve_sigma_shifts(guider)
    scales = config.scales
    if len(scales) < 2:
        raise ValueError("need at least two stages (scales ending at 1.0)")

    full_t, full_h, full_w = full_video.shape[-3:]
    stage_shapes = [
        stage_resolution(config, index, full_h, full_w, full_t)
        for index in range(len(scales))
    ]
    transition_steps = resolve_transition_steps(config, sigmas, full_h, full_w)
    if len(transition_steps) != len(scales) - 1:
        raise ValueError("transition steps count must be n_scales - 1")
    if any(not 0 < step < len(sigmas) - 1 for step in transition_steps):
        raise ValueError("transition step must be inside the sigma schedule")

    working_sigmas = sigmas.clone()
    stage_h, stage_w, stage_t = stage_shapes[0]
    coarse_video = full_video.new_zeros(
        full_video.shape[:-3] + (stage_t, stage_h, stage_w)
    )
    stage_start_latent = pack_latent(coarse_video, torch.zeros_like(full_audio))

    full_noise = None
    full_noise_video = None
    if config.noise_policy == "coupled_full_grid":
        full_noise = noise.generate_noise(latent)
        full_noise_video, full_noise_audio = unpack_latent(full_noise)
        coarse_noise_video = lowpass_dct(
            full_noise_video[..., :stage_t, :, :],
            (stage_h, stage_w),
        )
        stage_start_pub = pack_latent(coarse_noise_video, full_noise_audio)
    else:
        coarse_latent = latent.copy()
        coarse_latent["samples"] = stage_start_latent
        stage_start_pub = noise.generate_noise(coarse_latent)

    if x0_output is None:
        x0_output = {}
    global_total = len(sigmas) - 1
    stock_callback = preview_callback
    if stock_callback is None:
        stock_callback = _build_preview_callback(guider, global_total, x0_output)

    if sampler_override is not None:
        if sampler_name != "euler":
            raise ValueError(
                "Pass either sampler_name or sampler_override, not both "
                f"(got sampler_name={sampler_name!r} and sampler_override)."
            )
        sampler_handle = _OverrideSamplerHandle(sampler_override)
    else:
        sampler_handle = create_speed_sampler_handle(sampler_name)

    walker = LatentWalker(guider)
    global_start = 0
    global_done = 0
    last_public = None

    try:
        for stage_idx, global_end in enumerate(transition_steps):
            stage_h, stage_w, _ = stage_shapes[stage_idx]
            walker.apply_stage(stage_h, stage_w)

            stage_sigmas = working_sigmas[global_start : global_end + 1]
            callback = _wrap_preview_callback(
                stock_callback,
                x0_output,
                global_done,
                global_total,
            )
            public = guider.sample(
                stage_start_pub,
                stage_start_latent,
                sampler_handle.sampler,
                stage_sigmas,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
            last_public = public
            global_done += len(stage_sigmas) - 1

            boundary_sigma = float(working_sigmas[global_end])
            public_video, public_audio = unpack_latent(public)
            internal_video, internal_audio = to_internal_state(
                public_video,
                public_audio,
                boundary_sigma,
                audio_scale,
            )

            ratio = scales[stage_idx + 1] / scales[stage_idx]
            if config.sigma_policy == "canonical":
                kappa, next_sigma = aligned_sigma(boundary_sigma, ratio)
            else:
                kappa, next_sigma = 1.0, boundary_sigma
            working_sigmas[global_end] = next_sigma

            next_h, next_w, next_t = stage_shapes[stage_idx + 1]
            expanded_video = _expand_video(
                internal_video,
                target_thw=(next_t, next_h, next_w),
                sigma=boundary_sigma,
                seed=int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                full_noise_video=full_noise_video,
            )
            transitioned_video = expanded_video * kappa

            old_audio_sigma = time_shift_sigma(
                boundary_sigma,
                video_shift,
                audio_shift,
            )
            new_audio_sigma = time_shift_sigma(
                next_sigma,
                video_shift,
                audio_shift,
            )
            if config.audio_policy == "carry_preserve":
                transitioned_audio = carry_preserved_audio(
                    internal_audio,
                    boundary_sigma,
                    next_sigma,
                    old_audio_sigma,
                    new_audio_sigma,
                )
            elif config.audio_policy == "clock_reindex":
                if "x0" not in x0_output:
                    raise RuntimeError("clock_reindex requires an x0 callback from this stage")
                _, clean_audio = unpack_latent(x0_output["x0"])
                transitioned_audio = clock_reindex_audio_state(
                    internal_audio,
                    clean_audio,
                    boundary_sigma,
                    next_sigma,
                    old_audio_sigma,
                    new_audio_sigma,
                    audio_scale,
                )
            else:
                transitioned_audio = internal_audio

            sampler_handle.on_transition(
                SpeedTransition(
                    stage_idx=stage_idx,
                    ratio=ratio,
                    old_sigma=boundary_sigma,
                    new_sigma=next_sigma,
                    source_thw=tuple(internal_video.shape[-3:]),
                    target_thw=(next_t, next_h, next_w),
                )
            )

            stage_start_pub = pack_latent(
                reentry_noise(transitioned_video, next_sigma),
                reentry_noise(transitioned_audio, next_sigma),
            )
            stage_start_latent = pack_latent(
                torch.zeros_like(transitioned_video),
                torch.zeros_like(transitioned_audio),
            )
            global_start = global_end

        walker.apply_final()
        final_sigmas = working_sigmas[global_start:]
        final_callback = _wrap_preview_callback(
            stock_callback,
            x0_output,
            global_done,
            global_total,
        )
        last_public = guider.sample(
            stage_start_pub,
            stage_start_latent,
            sampler_handle.sampler,
            final_sigmas,
            callback=final_callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )

        if output_device is not None:
            last_public = last_public.to(output_device)

        output = latent.copy()
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        output["samples"] = last_public

        denoised = output
        x0 = x0_output.get("x0")
        if x0 is not None:
            denoised = latent.copy()
            denoised["samples"] = guider.model_patcher.model.process_latent_out(
                x0.cpu() if hasattr(x0, "cpu") else x0
            )
        return output, denoised
    finally:
        try:
            sampler_handle.close()
        finally:
            walker.apply_final()
