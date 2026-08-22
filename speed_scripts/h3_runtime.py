"""MiniMax-H3 SPEED stage runner — self-contained correctness oracle.

Wraps each SPEED stage in a separate `guider.sample()` call so the H3 model
always sees a buffer matching its latent_shapes. Ported from the Lab's
`h3_runtime.py`.
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
    to_internal_state,
    reentry_noise,
    time_shift_sigma,
)
from .spectral import (
    dct2, idct2, idct_temporal, lowpass_dct,
    spectral_expand, spectral_expand_3d, spectral_expand_coupled,
    dct_temporal,
)

log = logging.getLogger(__name__)


def _downscale_cond_latents(payload, stage_h, stage_w, is_final_stage=False):
    """[Level 3] Downscale cond_video_latents to (stage_h, stage_w) if they exceed current stage dims.

    Called by: `_patch_guider_payload_for_stage` (the per-stage I2V fix).
    Mutates payload dict in-place so the model sees condition latents matching
    the coarse latent resolution.
    
    Args:
        payload: The minimax_payload dict containing cond_video_latents
        stage_h: Target height for this stage
        stage_w: Target width for this stage
        is_final_stage: If True, skip downscale (latents already at full resolution)
    """
    conds = payload.get("cond_video_latents", [])
    if not conds:
        if is_final_stage:
            log.info("[SPEED] Final stage - cond_video_latents already at correct resolution, no need to check")
        return
    if is_final_stage:
        log.info("[SPEED] Final stage - cond_video_latents already at correct resolution, skipping downscale")
        return
    for i, z in enumerate(conds):
        z_h, z_w = z.shape[-2], z.shape[-1]
        if z_h > stage_h or z_w > stage_w:
            conds[i] = torch.nn.functional.interpolate(
                z, size=(stage_h, stage_w), mode="bilinear", align_corners=False,
            )


def _wrap_model_cond_video_rows(model, stage_h, stage_w):
    """[Level 3] Monkey-patch model._cond_video_rows to downscale conditions to stage resolution.
    
    This guarantees the I2V condition latents match the current stage's coarse
    resolution, fixing the shape mismatch in all_video_rows[~img_update] = cond_video_rows.
    
    Args:
        model: The H3 model whose _cond_video_rows method we wrap.
        stage_h: Target height for this stage.
        stage_w: Target width for this stage.
    """
    if not hasattr(model, "_cond_video_rows"):
        # T2V models have no I2V condition rows; the wrap would crash.
        log.info("[SPEED-MONKEY] model %s has no _cond_video_rows — skipping wrap",
                 type(model).__name__)
        return
    original_method = model._cond_video_rows
    wrap_id = id(original_method)
    
    def _patched_cond_video_rows(payload, device, target_h=None, target_w=None):
        log.warning("[SPEED-MONKEY] _cond_video_rows CALLED wrap_id=%s target_h=%s target_w=%s payload_keys=%s",
                    wrap_id, target_h, target_w, list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)
        # Downscale condition latents to match stage resolution
        payload = payload or {}
        conds = payload.get("cond_video_latents", [])
        if conds:
            log.warning("[SPEED-MONKEY] Found %d cond latents before downscale", len(conds))
            for i, z in enumerate(conds):
                z_h, z_w = z.shape[-2], z.shape[-1]
                log.warning("[SPEED-MONKEY] cond[%d] shape=%s stage=(%d,%d) needs_downscale=%s",
                            i, list(z.shape), stage_h, stage_w, (z_h != stage_h or z_w != stage_w))
                if z_h != stage_h or z_w != stage_w:
                    conds[i] = torch.nn.functional.interpolate(
                        z.float(), size=(stage_h, stage_w),
                        mode="bilinear", align_corners=False,
                    )
                    log.warning("[SPEED-MONKEY] cond[%d] downscaled to %s", i, list(conds[i].shape))
        else:
            log.warning("[SPEED-MONKEY] No cond_video_latents in payload")
        result = original_method(payload, device)
        log.warning("[SPEED-MONKEY] _cond_video_rows returned shape=%s", list(result.shape) if hasattr(result, 'shape') else type(result).__name__)
        return result
    
    model._cond_video_rows = _patched_cond_video_rows
    log.warning("[SPEED-MONKEY] Monkey-patched _cond_video_rows model=%s stage=(%d,%d) wrap_id=%s",
                type(model).__name__, stage_h, stage_w, wrap_id)


def _patch_guider_payload_for_stage(guider, stage_h, stage_w, stage_t, audio_t, is_final_stage=False):
    """[Level 3] Per-stage I2V fix.

    The condition latents (I2V reference image / keyframes) are full-resolution
    [B,C,T,H_full,W_full]. SPEED runs each stage at a *coarser* resolution, so
    the model builds its PackedLayout from the coarse latent dims. If we leave
    the condition latents at full resolution, `_cond_video_rows` produces 4x more
    rows than `layout.img_update` allocated -> broadcast shape mismatch at
    `model.py: all_video_rows[~img_update] = cond_video_rows`.

    Fix (lives in the extension, not base ComfyUI): walk the guider's original_conds,
    find the `minimax_payload` CONDConstant, downscale `cond_video_latents` to the
    current stage dims. The model at model.py:520-524 auto-rebuilds its PackedLayout
    when it detects a signature mismatch, producing the correct img_update rows.

    MiniMax has no negative prompts, so only the positive cond is patched.

    Args:
        guider: The guider object containing original_conds
        stage_h: Target height for this stage
        stage_w: Target width for this stage
        stage_t: Target temporal dimension
        audio_t: Audio temporal dimension
        is_final_stage: If True, skip downscale (latents already at full resolution)
    """
    model = guider.model_patcher.model
    # Guider_Basic/CFGGuider only creates `self.conds` inside inner_sample,
    # copying from original_conds. Patch original_conds so each stage's
    # guider.sample() picks up the downscaled payload.
    
    # Handle both dict and non-dict formats for original_conds
    original_conds = getattr(guider, 'original_conds', {})
    if not isinstance(original_conds, dict):
        original_conds = {}
    
    positive_conds = original_conds.get("positive", [])
    if not isinstance(positive_conds, list):
        positive_conds = []
    
    for cond in positive_conds:
        payload_holder = None
        
        # Handle both dict and object (CONDConstant) formats for cond
        if isinstance(cond, dict):
            payload_holder = cond.get("minimax_payload")
        else:
            # Try attribute access for CONDConstant objects
            payload_holder = getattr(cond, "minimax_payload", None)
        
        if payload_holder is None:
            continue
        
        # CONDConstant wraps the raw dict in `.cond`
        payload = getattr(payload_holder, "cond", None)
        
        # If we couldn't get it via .cond, try using payload_holder directly
        if payload is None and isinstance(payload_holder, dict):
            payload = payload_holder
        
        if not isinstance(payload, dict):
            continue
        
        _downscale_cond_latents(payload, stage_h, stage_w, is_final_stage=is_final_stage)


def stage_resolution(config, stage_idx, full_h, full_w, full_t):
    """[Level 2] Resolve the (h, w, t) a given stage runs at.

    Called by: `run_speed_pipeline` (the main pipeline orchestrator) at each
    stage iteration and at the final stage. Returns the coarse spatial/temporal
    dimensions that SPEED stage `stage_idx` will operate at.
    """
    scales = config.scales
    s = scales[stage_idx]
    h = max(1, round(full_h * s))
    w = max(1, round(full_w * s))
    if config.temporal_scales and stage_idx < len(config.temporal_scales):
        t = max(1, round(full_t * config.temporal_scales[stage_idx]))
    else:
        t = full_t
    return h, w, t


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def power_at_frequency(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum P(omega) = A * |omega|^(-beta). Matches paper Eq. 8."""
    return A * abs(omega) ** (-beta)


def activation_threshold(P_omega: float, delta: float) -> float:
    """Activation time for one radial frequency. Matches paper Eq. 9."""
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unpack_latent(samples):
    """[Level 2] Unpack a NestedTensor into (video, audio) with H3 geometry validation."""
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
    """[Level 2] Pack (video, audio) into a NestedTensor."""
    from comfy import nested_tensor as default_comfy_nested_tensor
    return default_comfy_nested_tensor.NestedTensor([video, audio])


def resolve_sigma_shifts(guider):
    """[Level 2] Return (video_shift, audio_shift, audio_scale) from the guider's model.

    Resolves in priority order:
        transformer_options['minimax_h3_sigma_shift_video/audio']
        -> model.sigma_shift_video/audio
        -> model.diffusion_model.sigma_shift_video/audio
    audio_scale is the constant bridge ratio used by flow.to_internal_state.
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


def _step_capture():
    """[Level 2] Build a step callback that records per-step state and drives the
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
    """[Level 3] First index whose sigma <= threshold; len-1 if none."""
    vals = [float(s) for s in sigmas]
    n = len(vals) - 1
    for i in range(n):
        if vals[i] <= threshold:
            return i
    return n


def resolve_transition_steps(
    config: SpeedConfig, sigmas, H_full: int | None = None, W_full: int | None = None,
) -> tuple[int, ...]:
    """[Level 2] Resolve per-stage transition steps.

    Uses delta-optimal power-spectrum thresholds when the config requests it;
    otherwise falls back to the explicit transition_steps in the config.
    """
    scales = config.scales
    if config.transition_mode == "delta_custom":
        tolerance = config.delta
        A, beta = config.noise_amplitude, config.noise_decay_exponent
        if H_full is None or W_full is None:
            H_full, W_full = config.full_latent_h, config.full_latent_w
        steps = []
        for i in range(len(scales) - 1):
            omega_i = scales[i] * min(H_full, W_full) / 2.0
            p = power_at_frequency(omega_i, A, beta)
            thr = activation_threshold(p, tolerance)
            steps.append(_find_first_step_below(sigmas, thr))
        return tuple(steps)
    return tuple(int(s) for s in config.transition_steps)


def run_speed_pipeline(
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
    """[Level 1] Run an N-stage progressive-resolution Euler chain (multi-stage SPEED).

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
    full_video, full_audio = unpack_latent(samples)
    video_shift, audio_shift, audio_scale = resolve_sigma_shifts(guider)
    log.info("[SPEED] incoming latent  video=%s audio=%s nonzero_video=%s nonzero_audio=%s",
             list(full_video.shape), list(full_audio.shape),
             torch.count_nonzero(full_video).item(), torch.count_nonzero(full_audio).item())
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
    coarse_samples = pack_latent(
        coarse_video,
        torch.zeros_like(full_audio),
    )
    log.info("[SPEED] coarse stage 0  target_shape=%s (full=%s scale=%s)",
             list(coarse_video.shape), list(full_video.shape), scales[0])
    cur_latent = latent.copy()
    cur_latent["samples"] = coarse_samples
    
    # I2V fix: wrap model's _cond_video_rows to downscale to coarse stage dims
    log.warning("[SPEED-MONKEY] STAGE 0 SETUP: Wrapping _cond_video_rows")
    if hasattr(guider, 'model_patcher') and guider.model_patcher is not None:
        model = guider.model_patcher.model
        log.warning("[SPEED-MONKEY] Found model: %s - wrapping _cond_video_rows", type(model).__name__)
        _wrap_model_cond_video_rows(model, s0_h, s0_w)
    else:
        log.warning("[SPEED-MONKEY] NO model_patcher found - cannot wrap coarse stage 0")

    full_noise = None
    if config.noise_policy == "coupled_full_grid":
        full_noise = noise.generate_noise(latent)
        full_noise_video, full_noise_audio = unpack_latent(full_noise)
        # Apply temporal crop first (3D lowpass-like), then spatial DCT lowpass.
        coarse_noise_video = lowpass_dct(
            full_noise_video[..., :s0_t, :, :], (s0_h, s0_w)
        )
        coarse_noise = pack_latent(
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

        log.info("[SPEED] stage %d start: latent=%s pub=%s sigmas=%d boundary=%d",
                 stage_idx,
                 list(stage_start_latent.shape) if hasattr(stage_start_latent, 'shape') else stage_start_latent,
                 list(stage_start_pub.shape) if hasattr(stage_start_pub, 'shape') else stage_start_pub,
                 len(current_sigmas), boundary)

        # I2V per-stage fix: downscale cond_video_latents so they match the coarse
        # latent this stage actually runs at.
        sh, sw, st = stage_resolution(config, stage_idx, full_h, full_w, full_t)
        _patch_guider_payload_for_stage(guider, sh, sw, st, full_audio.shape[-1], is_final_stage=False)
        
        # Also wrap model's _cond_video_rows to ensure conditions match stage resolution
        log.warning("[SPEED-MONKEY] INTERIOR STAGE %d: wrapping _cond_video_rows", stage_idx)
        if hasattr(guider, 'model_patcher') and guider.model_patcher is not None:
            model = guider.model_patcher.model
            log.warning("[SPEED-MONKEY] Found model: %s - wrapping _cond_video_rows", type(model).__name__)
            _wrap_model_cond_video_rows(model, sh, sw)
        else:
            log.warning("[SPEED-MONKEY] NO model_patcher found - cannot wrap interior stage %d", stage_idx)

        # Run the current stage over current_sigmas[:boundary+1].
        capture, callback = _step_capture()
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
        pub_vid, pub_aud = unpack_latent(public)
        log.info("[SPEED] stage %d output: video=%s audio=%s q=%.4f",
                 stage_idx, list(pub_vid.shape), list(pub_aud.shape), float(current_sigmas[boundary]))

        public_video, public_audio = unpack_latent(public)
        q = float(current_sigmas[boundary])

        # Recover internal state (public -> carry-representation).
        internal_video, internal_audio = to_internal_state(
            public_video, public_audio, q, audio_scale
        )

        # Align (kappa) for this transition: r = next_scale / current_scale.
        ratio = scales[stage_idx + 1] / scales[stage_idx]
        if config.sigma_policy == "canonical":
            kappa, new_q = aligned_sigma(q, ratio)
        else:
            kappa, new_q = 1.0, q

        # DCT-expand the video (coupled or fresh band) and rescale by kappa.
        next_hw = stage_hw[stage_idx + 1]
        next_t = stage_t[stage_idx + 1]
        if next_t > internal_video.shape[-3]:
            # Temporal expansion needed: use the 3D spectral path.
            if config.noise_policy == "coupled_full_grid":
                full_noise_video, _ = unpack_latent(full_noise)
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
                expanded_video = spectral_expand_3d(
                    internal_video,
                    (next_t, *next_hw),
                    q,
                    int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                )
        elif config.noise_policy == "coupled_full_grid":
            full_noise_video, _ = unpack_latent(full_noise)
            expanded_video = spectral_expand_coupled(
                internal_video,
                full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype),
                q,
            )
        else:
            expanded_video = spectral_expand(
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
            transitioned_audio = carry_preserved_audio(
                internal_audio, q, new_q, old_audio_sigma, new_audio_sigma
            )
        elif config.audio_policy == "clock_reindex":
            if "x0" not in capture:
                raise RuntimeError("clock_reindex requires an x0 callback from this stage")
            _, clean_audio = unpack_latent(capture["x0"])
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
        next_noise = pack_latent(
            reentry_noise(transitioned_video, new_q),
            reentry_noise(transitioned_audio, new_q),
        )
        next_zero = pack_latent(
            torch.zeros_like(transitioned_video),
            torch.zeros_like(transitioned_audio),
        )
        log.info("[SPEED] stage %d → %d: expanded=%s next_zero=%s new_q=%.4f",
                 stage_idx, stage_idx + 1,
                 list(transitioned_video.shape) if hasattr(transitioned_video, "shape") else transitioned_video,
                 list(next_zero.shape) if hasattr(next_zero, "shape") else next_zero,
                 new_q)

        # Advance to the next stage.
        stage_start_pub = next_noise
        stage_start_latent = next_zero
        current_sigmas = next_sigmas

    # After the final transition, run the last full-res stage over the spliced tail.
    log.info("[SPEED] final stage: latent=%s sigmas=%d",
             list(stage_start_latent.shape) if hasattr(stage_start_latent, 'shape') else stage_start_latent,
             len(current_sigmas))
    # Final stage is at scale 1.0 (stage n_stages-1) so target == full res -> no-op
    # downscale/rebuild, keeping T2V and full-res I2V behaviour unchanged.
    fh, fw, ft = stage_resolution(config, n_stages - 1, full_h, full_w, full_t)
    _patch_guider_payload_for_stage(guider, fh, fw, ft, full_audio.shape[-1], is_final_stage=True)
    final_capture, final_callback = _step_capture()
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