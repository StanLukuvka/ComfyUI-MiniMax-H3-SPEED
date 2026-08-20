"""MiniMax-H3 SPEED stage runner — self-contained correctness oracle.

Wraps each SPEED stage in a separate `guider.sample()` call so the H3 model
always sees a buffer matching its latent_shapes. Ported from the Lab's
`h3_runtime.py`.
"""

from __future__ import annotations

import math

import torch

from .config import SpeedConfig
from .flow import (
    aligned_speed_sigma,
    carry_preserving_audio_state,
    clock_reindex_audio_state,
    recover_internal_state,
    reentry_noise,
    time_shift_sigma,
)
from .spectral import (
    dct2, idct2, idct_temporal, lowpass_dct,
    spectral_expand_dct, spectral_expand_dct_3d, spectral_expand_dct_coupled,
    dct_temporal,
)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum P(omega) = A * |omega|^(-beta). Matches paper Eq. 8."""
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Activation time for one radial frequency. Matches paper Eq. 9."""
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_tensor(samples):
    """Unpack a NestedTensor into (video, audio) with H3 geometry validation."""
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


def _pack_tensor(video, audio):
    """Pack (video, audio) into a NestedTensor."""
    from comfy import nested_tensor as default_comfy_nested_tensor
    return default_comfy_nested_tensor.NestedTensor([video, audio])


def _active_av_shifts(guider):
    """Return (video_shift, audio_shift, audio_scale) from the guider's model.

    ComfyUI exposes the active shifts through ModelSamplingAV. The diffusion
    model keeps the checkpoint defaults, while transformer options may contain
    overrides from the MiniMaxH3SigmaShift node.
    audio_scale is the constant bridge ratio used by flow.recover_internal_state.
    """
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None)
    if model is None:
        raise ValueError("no model_patcher.model on guider")

    # PR3 (audio-fix): Candidate priority order MUST match the H3 model's own
    # resolution logic (model.py:527): explicit transformer_options overrides,
    # then the model's own sigma_shift_video/audio attributes (default 12.0/3.0).
    # Assume if the H3-specific 12.0/3.0 default is not present then an error
    # occurs — do NOT fall back to ComfyUI's generic ModelSamplingAV.shift
    # (a different quantity, flow-matching shift often 1.0), or audio_scale
    # collapses and every audio transition is rescaled ~12x wrong (garbled sound).
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

    # PR3 (audio-fix): H3 model's own authoritative sigma shifts (default 12.0 / 3.0). DO NOT TOUCH.
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

    shifts = next((pair for pair in candidates if all(isinstance(v, (int, float)) for v in pair)), None)
    if shifts is None:
        raise ValueError(
            "active MiniMax-H3 sigma shifts are unavailable: the loaded model "
            "does not expose sigma_shift_video/audio (on `model`, `model.diffusion_model`, "
            "or `model_options['transformer_options']['minimax_h3_sigma_shift_*']`). "
            "The MiniMax-H3 SPEED sampler requires a real MiniMax-H3 model; loading a "
            "non-H3 model (SD/Flux/WAN/etc.) is a configuration error."
        )
    video_shift, audio_shift = map(float, shifts)
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError("active MiniMax-H3 shifts must be positive")
    return video_shift, audio_shift, video_shift / audio_shift


def _capture():
    """Build a step callback that records per-step state and drives the
    ComfyUI UI progress bar (ProgressBar.update_absolute).

    ComfyUI's web UI bar is NOT drawn by k_diffusion's `disable=` flag — it's
    driven by `comfy.utils.ProgressBar(total).update_absolute(...)` called
    inside a node's execution context (latent_preview.prepare_callback does
    this for the official SamplerCustomAdvanced). Our sampler must do the
    same, otherwise the UI bar never appears regardless of PROGRESS_BAR_ENABLED.

    `total_steps` is forwarded by the k_diffusion callback wrapper and is the
    only reliable count we have at callback construction time.
    """
    state = {}

    try:
        import comfy.utils as _comfy_utils  # type: ignore
    except Exception:
        _comfy_utils = None

    pbar = _comfy_utils.ProgressBar(1) if _comfy_utils is not None else None

    def callback(step, x0, x, total_steps):
        state["x0"] = x0
        state["x"] = x
        state["step"] = step
        state["total_steps"] = total_steps
        if pbar is not None:
            pbar.update_absolute(step + 1, total_steps)

    return state, callback


def _find_first_step_below(sigmas, threshold: float) -> int:
    """First index whose sigma <= threshold; len-1 if none."""
    vals = [float(s) for s in sigmas]
    n = len(vals) - 1
    for i in range(n):
        if vals[i] <= threshold:
            return i
    return n


def resolve_transition_steps(
    config: SpeedConfig, sigmas, H_full: int | None = None, W_full: int | None = None,
) -> tuple[int, ...]:
    """Resolve per-stage transition steps.

    Uses delta-optimal power-spectrum thresholds when the config requests it;
    otherwise falls back to the explicit transition_steps in the config.
    """
    scales = config.scales
    if config.transition_mode == "delta_custom":
        tolerance = config.delta
        A, beta = config.power_A, config.power_beta
        if H_full is None or W_full is None:
            H_full, W_full = config.full_latent_h, config.full_latent_w
        steps = []
        for i in range(len(scales) - 1):
            omega_i = scales[i] * min(H_full, W_full) / 2.0
            p = power_spectrum(omega_i, A, beta)
            thr = activation_time(p, tolerance)
            steps.append(_find_first_step_below(sigmas, thr))
        return tuple(steps)
    return tuple(int(s) for s in config.transition_steps)


def run_repeated_stage_calls(
    noise,
    guider,
    sigmas: torch.Tensor,
    latent: dict,
    config: SpeedConfig,
    *,
    sampler,
    nested_type,
    # PR3 (progress-bar): KEEP THE PROGRESS BAR ON BY DEFAULT. disable_pbar
    # defaults to False (bar VISIBLE). The SPEED sampler node explicitly passes
    # `disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED` to honor the user's
    # ComfyUI setting. DO NOT change this default back to True — doing so hides
    # the progress bar for every run and is easy to miss (the node still "works").
    disable_pbar: bool = False,
    output_device=None,
):
    """Run an N-stage progressive-resolution Euler chain (multi-stage SPEED).

    This is intentionally the slow correctness oracle: each public guider call
    performs its own prepare/pre-run/cleanup lifecycle and naturally rebuilds
    H3's shape-dependent conditions.

    Mirrors canonical SPEED ``generate``: it computes transition steps from
    delta-optimal thresholds, runs each scale stage, and DCT-expands + kappa-aligns
    at each boundary.

    Returns ``(output_latent, denoised_latent)``.
    """
    if "noise_mask" in latent:
        raise ValueError("T2V oracle does not support noise masks")
    samples = latent.get("samples")
    full_video, full_audio = _unpack_tensor(samples)
    video_shift, audio_shift, audio_scale = _active_av_shifts(guider)
    if torch.count_nonzero(full_video) or torch.count_nonzero(full_audio):
        raise ValueError("T2V oracle currently requires an empty H3 latent")
    if sigmas.ndim != 1 or len(sigmas) < 3:
        raise ValueError("sigmas must be a one-dimensional schedule")

    scales = config.scales
    n_stages = len(scales)
    if n_stages < 2:
        raise ValueError("need at least two stages (scales ending at 1.0)")

    full_h, full_w = full_video.shape[-2:]
    full_t = full_video.shape[-3]
    stage_hw = [
        (max(1, round(full_h * s)), max(1, round(full_w * s))) for s in scales
    ]
    if config.temporal_scales:
        stage_t = [max(1, round(full_t * ts)) for ts in config.temporal_scales]
    else:
        stage_t = [full_t] * n_stages

    # Resolve transition steps (delta-optimal or explicit), using the LIVE full
    # resolution latent dims (matches canonical SPEED x.shape[-2:]) rather than
    # the planner's config defaults.
    transition_steps = resolve_transition_steps(config, sigmas, full_h, full_w)
    if len(transition_steps) != n_stages - 1:
        raise ValueError("transition steps count must be n_scales - 1")
    for ts in transition_steps:
        if not 0 < ts < len(sigmas) - 1:
            raise ValueError("transition step must be inside the sigma schedule")

    # Stage 1: initialize coarse latent + noise at scale[0].
    s0_h, s0_w = stage_hw[0]
    s0_t = stage_t[0]
    coarse_video = full_video.new_zeros(full_video.shape[:-3] + (s0_t, s0_h, s0_w))
    coarse_samples = _pack_tensor(
        coarse_video,
        torch.zeros_like(full_audio),
    )
    cur_latent = latent.copy()
    cur_latent["samples"] = coarse_samples

    full_noise = None
    if config.noise_policy == "coupled_full_grid":
        full_noise = noise.generate_noise(latent)
        full_noise_video, full_noise_audio = _unpack_tensor(full_noise)
        # Apply temporal crop first (3D lowpass-like), then spatial DCT lowpass.
        coarse_noise_video = lowpass_dct(
            full_noise_video[..., :s0_t, :, :], (s0_h, s0_w)
        )
        coarse_noise = _pack_tensor(
            coarse_noise_video,
            full_noise_audio,
        )
    else:
        coarse_noise = noise.generate_noise(cur_latent)

    # Canonical generate structure: each scale stage runs a sigma slice, and at
    # every boundary we DCT-expand + kappa-align, then re-enter with the aligned
    # boundary sigma (new_q) prepended to the remaining tail. `transition_steps`
    # (already resolved above, delta-optimal when transition_mode == delta_custom)
    # names the per-stage boundary index into the current schedule.

    # current_sigmas is the live schedule for the CURRENT stage: it starts as the
    # full input sigmas, and after each boundary is spliced to [new_q] + tail.
    current_sigmas = sigmas
    stage_start_pub = coarse_noise
    stage_start_latent = cur_latent["samples"]
    last_public = None
    last_capture = None

    for stage_idx in range(n_stages - 1):
        # Boundary for this stage: transition_steps[stage_idx] is an index into
        # current_sigmas identifying where the NEXT scale begins.
        boundary = int(transition_steps[stage_idx])
        # Guard against running past the schedule; clamp to interior.
        n_avail = len(current_sigmas) - 1
        boundary = min(boundary, n_avail - 1)
        if boundary < 1:
            raise ValueError("transition step must be inside the sigma schedule")

        # Run the current stage over current_sigmas[:boundary+1].
        capture, callback = _capture()
        stage_sigmas = current_sigmas[: boundary + 1]
        public = guider.sample(
            stage_start_pub,
            stage_start_latent,
            sampler,
            stage_sigmas,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        last_public = public
        last_capture = capture

        public_video, public_audio = _unpack_tensor(public)
        q = float(current_sigmas[boundary])

        # Recover internal state (public -> carry-representation).
        internal_video, internal_audio = recover_internal_state(
            public_video, public_audio, q, audio_scale
        )

        # Align (kappa) for this transition: r = next_scale / current_scale.
        ratio = scales[stage_idx + 1] / scales[stage_idx]
        if config.sigma_policy == "canonical":
            kappa, new_q = aligned_speed_sigma(q, ratio)
        else:
            kappa, new_q = 1.0, q

        # DCT-expand the video (coupled or fresh band) and rescale by kappa.
        next_hw = stage_hw[stage_idx + 1]
        next_t = stage_t[stage_idx + 1]
        if next_t > internal_video.shape[-3]:
            # Temporal expansion needed: use the 3D spectral path.
            if config.noise_policy == "coupled_full_grid":
                full_noise_video, _ = _unpack_tensor(full_noise)
                # For coupled 3D: re-DCT-expand using full noise + cropped source.
                # Combined low-freq block = DCT of source (3D); high-freq = scaled noise.
                full_noise_video_dev = full_noise_video.to(
                    device=internal_video.device, dtype=internal_video.dtype,
                )
                # Slice full noise to next_t in temporal axis, then use 3D coupled-style
                # expansion: source DCT coefs go in low-freq corner, full noise coefs elsewhere.
                source_dct = dct2(dct_temporal(internal_video))
                target_noise = full_noise_video_dev[..., :next_t, :, :]
                target_dct = dct2(dct_temporal(target_noise)) * float(q)
                target_dct[..., :internal_video.shape[-3], :internal_video.shape[-2], :internal_video.shape[-1]] = source_dct
                expanded_video = idct_temporal(idct2(target_dct))
            else:
                expanded_video = spectral_expand_dct_3d(
                    internal_video,
                    (next_t, *next_hw),
                    q,
                    int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                )
        elif config.noise_policy == "coupled_full_grid":
            full_noise_video, _ = _unpack_tensor(full_noise)
            expanded_video = spectral_expand_dct_coupled(
                internal_video,
                full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype),
                q,
            )
        else:
            expanded_video = spectral_expand_dct(
                internal_video,
                next_hw,
                q,
                int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
            )
        transitioned_video = expanded_video * kappa

        # Audio handling at this boundary.
        old_audio_sigma = time_shift_sigma(q, video_shift, audio_shift)
        new_audio_sigma = time_shift_sigma(new_q, video_shift, audio_shift)
        if config.audio_policy == "carry_preserve":
            transitioned_audio = carry_preserving_audio_state(
                internal_audio, q, new_q, old_audio_sigma, new_audio_sigma
            )
        elif config.audio_policy == "clock_reindex":
            if "x0" not in capture:
                raise RuntimeError("clock_reindex requires an x0 callback from this stage")
            _, clean_audio = _unpack_tensor(capture["x0"])
            transitioned_audio = clock_reindex_audio_state(
                internal_audio,
                clean_audio,
                q,
                new_q,
                old_audio_sigma,
                new_audio_sigma,
                audio_scale,
            )
        else:
            transitioned_audio = internal_audio

        # Build the next-stage schedule: aligned boundary + remaining tail.
        next_sigmas = torch.cat(
            [current_sigmas.new_tensor([new_q]), current_sigmas[boundary + 1:]], dim=0
        )

        # Set up re-entry with the aligned boundary + zero latent for the next stage.
        next_noise = _pack_tensor(
            reentry_noise(transitioned_video, new_q),
            reentry_noise(transitioned_audio, new_q),
        )
        next_zero = _pack_tensor(
            torch.zeros_like(transitioned_video),
            torch.zeros_like(transitioned_audio),
        )

        # Advance to the next stage.
        stage_start_pub = next_noise
        stage_start_latent = next_zero
        current_sigmas = next_sigmas

    # After the final transition, run the last full-res stage over the spliced tail.
    final_capture, final_callback = _capture()
    final_public = guider.sample(
        stage_start_pub,
        stage_start_latent,
        sampler,
        current_sigmas,
        callback=final_callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    last_public = final_public
    last_capture = final_capture

    if output_device is not None and last_public is not None:
        last_public = last_public.to(output_device)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = last_public

    denoised = out
    if last_capture is not None and "x0" in last_capture:
        x0 = last_capture["x0"]
        # x0 may be a NestedTensor — extract video stream
        if getattr(x0, "is_nested", False):
            x0_streams = list(x0.unbind())
            x0_video = next((s for s in x0_streams if s.ndim == 5), None)
            if x0_video is not None:
                x0 = x0_video
        denoised = latent.copy()
        denoised["samples"] = guider.model_patcher.model.process_latent_out(
            x0.cpu() if hasattr(x0, "cpu") else x0
        )
    return out, denoised
