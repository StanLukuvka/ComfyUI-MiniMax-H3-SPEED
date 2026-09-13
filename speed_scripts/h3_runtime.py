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
    spectral_expand, spectral_expand_3d,
    dct_temporal,
)

log = logging.getLogger(__name__)

from .latent_class import LatentWalker
from .sampler_support import (
    SamplerCapability,
    SpeedTransition,
    SpeedSamplerHandle,
    create_speed_sampler_handle,
)


# Per-pipeline-run walker, stashed on the guider so the same wrapper dict
# survives every coarse stage and the final restore. Dropped at the end of
# every run; recreated on the next run that touches the same guider.
_LW_ATTR = "_speed_latent_walker"


class _OverrideSamplerHandle(SpeedSamplerHandle):
    """Test-seam handle: wraps an injected sampler object as stateless."""

    def __init__(self, sampler):
        self.sampler = sampler
        self.capability = SamplerCapability.STATELESS_STEP_LOCAL


def _get_or_create_walker(guider) -> LatentWalker:
    """Return the walker stashed on this guider, creating it on first use."""
    lw_existing = getattr(guider, _LW_ATTR, None)
    if lw_existing is not None:
        return lw_existing
    lw_walker = LatentWalker(guider)
    try:
        setattr(guider, _LW_ATTR, lw_walker)
    except (AttributeError, TypeError):
        # Some guider mocks refuse setattr; fall back to local-only walker.
        pass
    return lw_walker


def _drop_walker(guider) -> None:
    """Drop the walker at the end of a run. Pristine clones die with it."""
    lw_walker = getattr(guider, _LW_ATTR, None)
    if lw_walker is not None:
        try:
            delattr(guider, _LW_ATTR)
        except (AttributeError, TypeError):
            pass


def stage_resolution(config, stage_idx, full_h, full_w, full_t):
    """[Level 2] Resolve the (h, w, t) a given stage runs at.

    Called by: `run_speed_pipeline` (the main pipeline orchestrator) at each
    stage iteration and at the final stage. Returns the coarse spatial/temporal
    dimensions that SPEED stage `stage_idx` will operate at.
    """
    sr_scales = config.scales
    sr_s = sr_scales[stage_idx]
    sr_h = max(1, round(full_h * sr_s))
    sr_w = max(1, round(full_w * sr_s))
    if config.temporal_scales and stage_idx < len(config.temporal_scales):
        sr_t = max(1, round(full_t * config.temporal_scales[stage_idx]))
    else:
        sr_t = full_t
    return sr_h, sr_w, sr_t

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

def _build_preview_callback(guider, total_steps, x0_output):
    """Forward ComfyUI's native `latent_preview.prepare_callback` pipeline.

    Mirrors `SamplerCustomAdvanced.execute` (comfy_extras/nodes_custom_sampler.py
    ~1045-1049): one `x0_output` dict shared across the whole run, one callback
    built for the full step total. The callback writes `x0_output["x0"]` every
    step and pushes preview bytes via the same `PROGRESS_BAR_HOOK` ComfyUI uses
    everywhere else.

    Returns None when previews are unavailable (no patcher, import error, or
    no decoder). Callers should pass the returned callback into each
    `guider.sample()` call wrapped with `_wrap_preview_callback` so the bar's
    `step+1` value advances continuously across stages instead of resetting.
    """
    try:
        import latent_preview as _lp
    except Exception as exc:
        log.info("[SPEED-preview] latent_preview unavailable (%r) — previews disabled", exc)
        return None
    # Mirrors SamplerCustomAdvanced (comfy_extras/nodes_custom_sampler.py:1045-1046):
    # the patcher must be valid; if it is not, let the exception propagate.
    return _lp.prepare_callback(guider.model_patcher, total_steps, x0_output)


def _wrap_preview_callback(stock_cb, capture_state, global_offset, global_total):
    """Wrap a stock `latent_preview.prepare_callback` callback for one stage.

    Each `guider.sample()` call gets a fresh callback whose local `step`
    resets to 0. We record per-step state (for `clock_reindex` audio) and
    forward the call to the shared stock callback with the step number
    remapped to the global timeline, so the underlying `ProgressBar` runs
    continuously `0..global_total` instead of resetting each stage.

    `stock_cb` is the closure returned by `latent_preview.prepare_callback`;
    its own `(step, x0, x, total_steps)` call writes `x0_output["x0"] = x0`
    and calls `pbar.update_absolute(step + 1, total_steps, preview_bytes)`.
    We pass `global_offset + step` so the bar sees a monotonically increasing
    value, with `total_steps` overridden to `global_total` so its throttle
    uses the full run length.

    When `stock_cb is None` (previews disabled in the env) we still build a
    plain progress bar via `comfy.utils.ProgressBar` so the bar keeps moving.
    """
    state = capture_state
    if stock_cb is None:
        try:
            import comfy.utils as _cu
            pbar = _cu.ProgressBar(global_total)
        except Exception:
            pbar = None
        def callback(step, x0, x, total_steps):
            state["x0"] = x0
            if pbar is not None:
                try:
                    pbar.update_absolute(global_offset + step + 1, global_total)
                except Exception:
                    pass
        return callback
    warned = False
    def callback(step, x0, x, total_steps):
        nonlocal warned
        state["x0"] = x0
        try:
            stock_cb(global_offset + step, x0, x, global_total)
        except Exception as exc:
            # First failure must be loud; repeats would spam per step and
            # would also mask the real exception that broke the run.
            if not warned:
                warned = True
                log.warning(
                    "[SPEED-preview] preview callback failed (%r) — "
                    "preview image and bar updates will be missing for the rest of this run",
                    exc,
                )
    return callback

def _find_first_step_below(sigmas, threshold: float) -> int:
    """[Level 3] First index whose sigma <= threshold; len-1 if none."""
    ffsb_vals = [float(s) for s in sigmas]
    ffsb_n = len(ffsb_vals) - 1
    for ffsb_i in range(ffsb_n):
        if ffsb_vals[ffsb_i] <= threshold:
            return ffsb_i
    return ffsb_n

def resolve_transition_steps(
    config: SpeedConfig, sigmas, H_full: int | None = None, W_full: int | None = None,
) -> tuple[int, ...]:
    """[Level 2] Resolve per-stage transition steps.

    Uses delta-optimal power-spectrum thresholds when the config requests it;
    otherwise falls back to the explicit transition_steps in the config.
    """
    rts_scales = config.scales
    if config.transition_mode == "delta_custom":
        rts_tolerance = config.delta
        rts_A, rts_beta = config.noise_amplitude, config.noise_decay_exponent
        if H_full is None or W_full is None:
            H_full, W_full = config.full_latent_h, config.full_latent_w
        rts_steps = []
        for rts_i in range(len(rts_scales) - 1):
            rts_omega = rts_scales[rts_i] * min(H_full, W_full) / 2.0
            rts_p = power_at_frequency(rts_omega, rts_A, rts_beta)
            rts_thr = activation_threshold(rts_p, rts_tolerance)
            rts_steps.append(_find_first_step_below(sigmas, rts_thr))
        return tuple(rts_steps)
    return tuple(int(s) for s in config.transition_steps)

def run_speed_pipeline(
    noise,
    guider,
    sigmas: torch.Tensor,
    latent: dict,
    config: SpeedConfig,
    *,
    sampler_name: str = "euler",
    # Test/programmatic seam only: injects a fake sampler object without
    # building a real Comfy sampler. Production node code must never use it;
    # it may not be combined with a non-default sampler_name.
    sampler_override=None,
    # PR3 (progress-bar): KEEP THE PROGRESS BAR ON BY DEFAULT. disable_pbar
    # defaults to False (bar VISIBLE). The SPEED sampler node explicitly passes
    # `disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED` to honor the user's
    # ComfyUI setting. DO NOT change this default back to True — doing so hides
    # the progress bar for every run and is easy to miss (the node still "works").
    disable_pbar: bool = False,
    output_device=None,
    preview_callback=None,
    x0_output=None,
):
    """[Level 1] Run an N-stage progressive-resolution sampling chain (multi-stage SPEED).

    This is intentionally the slow correctness oracle: each public guider call
    performs its own prepare/pre-run/cleanup lifecycle and naturally rebuilds
    H3's shape-dependent conditions.

    Mirrors canonical SPEED ``generate``: it computes transition steps from
    delta-optimal thresholds, runs each scale stage, and DCT-expands + kappa-aligns
    at each boundary.

    The sampler arrives through a run-scoped ``SpeedSamplerHandle``: built
    once from ``sampler_name`` (or wrapped from a test-only
    ``sampler_override``) before the stage loop, notified through
    ``on_transition()`` once per configured transition, and closed in the
    run-level cleanup.

    Live previews are forwarded from ComfyUI's `latent_preview.prepare_callback`
    pipeline (the same one `SamplerCustomAdvanced` uses). Pass a callback built
    by the node layer in `preview_callback`; if `None`, the function builds one
    for the current run via `_build_preview_callback` so previews keep working
    in offline tests and programmatic use. The shared `x0` dict (`x0_output`)
    is also forwarded so `denoised` is reconstructed exactly as in
    `SamplerCustomAdvanced`.

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
    # One resolution source — stage_resolution owns this math; the cond-patching
    # path also calls it, so a second inline copy here could silently diverge.
    stage_hw_t = [
        stage_resolution(config, i, full_h, full_w, full_t)
        for i in range(n_stages)
    ]

    # Resolve transition steps (delta-optimal or explicit), using the LIVE full
    # resolution latent dims (matches canonical SPEED x.shape[-2:]) rather than
    # the planner's config defaults.
    transition_steps = resolve_transition_steps(config, sigmas, full_h, full_w)
    if len(transition_steps) != n_stages - 1:
        raise ValueError("transition steps count must be n_scales - 1")
    for rsp_ts in transition_steps:
        if not 0 < rsp_ts < len(sigmas) - 1:
            raise ValueError("transition step must be inside the sigma schedule")
    # NOTE: resolved steps are NOT required to be strictly increasing. Upstream
    # SPEED lets multiple transitions quantize onto the same sigma index: the
    # repeated boundary yields a zero-denoising-step intermediate segment while
    # every transition still runs its spectral expand + alignment.

    # Upstream's working-sigmas model: slice every stage by GLOBAL boundary
    # indices into this working copy, and after each transition patch the
    # boundary coordinate in place with the aligned sigma (upstream:
    # `scheduler.sigmas[end] = t_tilde`). For unique boundaries this is
    # identical to replacing the boundary entry per stage slice; for coincident
    # boundaries the second transition reads the already-aligned coordinate and
    # aligns it again.
    working_sigmas = sigmas.clone()

    # Stage 1: initialize coarse latent + noise at scale[0].
    s0_h, s0_w, s0_t = stage_hw_t[0]
    coarse_video = full_video.new_zeros(full_video.shape[:-3] + (s0_t, s0_h, s0_w))
    coarse_samples = pack_latent(
        coarse_video,
        torch.zeros_like(full_audio),
    )
    log.info("[SPEED] coarse stage 0  target_shape=%s (full=%s scale=%s)",
             list(coarse_video.shape), list(full_video.shape), scales[0])
    cur_latent = latent.copy()
    cur_latent["samples"] = coarse_samples

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

    # Canonical generate structure: `transition_steps` (resolved above,
    # delta-optimal when transition_mode == "delta_custom") are GLOBAL indices
    # into the sigma schedule — never local indices into a shortened tail.
    # Stage k runs sigma indices [prev_boundary+1 .. boundary]
    # (stage 0: [0 .. boundary0]); at each boundary the output is DCT-expanded
    # + kappa-aligned, and the boundary coordinate in `working_sigmas` is
    # patched in place with the aligned sigma, so the next stage re-enters
    # at the aligned coordinate.
    stage_start_pub = coarse_noise
    stage_start_latent = cur_latent["samples"]
    last_public = None
    # Build ComfyUI's native preview callback once for the full run, mirroring
    # SamplerCustomAdvanced. The shared x0_output dict is used for denoised
    # reconstruction; the stock callback writes it every step and also pushes
    # preview bytes through PROGRESS_BAR_HOOK.
    global_total = len(sigmas) - 1
    if x0_output is None:
        x0_output = {}
    stock_cb = preview_callback
    if stock_cb is None:
        stock_cb = _build_preview_callback(guider, global_total, x0_output)
    global_done = 0
    global_start = 0  # GLOBAL index the next stage begins at (into working_sigmas).

    # One run-scoped sampler handle, built before the stage loop. The override
    # seam wraps a caller-supplied sampler object as a stateless no-op handle;
    # production paths go through the public sampler_name selector.
    if sampler_override is not None:
        if sampler_name != "euler":
            raise ValueError(
                "Pass either sampler_name or sampler_override, not both "
                f"(got sampler_name={sampler_name!r} and sampler_override)."
            )
        sampler_handle = _OverrideSamplerHandle(sampler_override)
    else:
        sampler_handle = create_speed_sampler_handle(sampler_name)
    # Run-scoped I2V lifecycle: the walker is created up front and EVERY exit
    # path (success, failure in a stage, transition, audio handling, spectral
    # expansion, or final sampling) passes through the finally block, which
    # restores pristine conditioning latents and removes `_speed_latent_walker`
    # from the guider. The guider never keeps a half-resized cond latent.
    walker = _get_or_create_walker(guider)
    try:
        for stage_idx in range(n_stages - 1):
            # Boundary for this stage: transition_steps[stage_idx] is a GLOBAL
            # index into the sigma schedule identifying where the NEXT scale
            # begins. The final stage's global_end is len(sigmas) - 1.
            global_end = int(transition_steps[stage_idx])

            log.info("[SPEED] stage %d start: latent=%s pub=%s sigmas=%d boundary=%d",
                     stage_idx,
                     list(stage_start_latent.shape) if hasattr(stage_start_latent, 'shape') else stage_start_latent,
                     list(stage_start_pub.shape) if hasattr(stage_start_pub, 'shape') else stage_start_pub,
                     global_end - global_start + 1, global_end)

            # I2V per-stage fix: rescale the STORED keyframe/ref latents in
            # guider.original_conds so the per-stage guider.sample() ->
            # process_conds -> model.extra_conds rebuild of minimax_payload
            # picks up cond_video_latents matching this stage's coarse latent.
            # (The payload dict from a previous stage is rebuilt from these
            # sources every call, so these sources are the only patch point.)
            rsp_sh, rsp_sw, _ = stage_resolution(config, stage_idx, full_h, full_w, full_t)
            walker.apply_stage(rsp_sh, rsp_sw)

            # Run the current stage over its global slice
            # (`working_sigmas[global_start : global_end + 1]`). The wrapped
            # callback forwards to the shared stock callback
            # (`latent_preview.prepare_callback`) with the step remapped to the
            # global timeline, so the bar runs continuously and preview bytes
            # flow through the same PROGRESS_BAR_HOOK every other ComfyUI node
            # uses. The wrapper writes x0 into the shared `x0_output` dict for
            # `clock_reindex` audio and the denoised fallback.
            preview_cb = _wrap_preview_callback(
                stock_cb, x0_output, global_done, global_total,
            )
            stage_sigmas = working_sigmas[global_start:global_end + 1]

            callback = preview_cb
            # H3-runtime behavior: a zero-step intermediate stage still invokes
            # guider.sample once with a single-sigma schedule (zero denoising
            # steps). Upstream SPEED skips the sampler for such segments.
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
            # Kappa alignment uses the transition sigma at the GLOBAL boundary
            # index of the working schedule — which previous transitions may have
            # already patched with an aligned coordinate.
            rsp_q = float(working_sigmas[global_end])
            public_video, public_audio = unpack_latent(public)
            log.info("[SPEED] stage %d output: video=%s audio=%s rsp_q=%.4f",
                     stage_idx, list(public_video.shape), list(public_audio.shape), rsp_q)

            # Recover internal state (public -> carry-representation).
            internal_video, internal_audio = to_internal_state(
                public_video, public_audio, rsp_q, audio_scale
            )

            # Align (kappa) for this transition: rsp_r = next_scale / current_scale.
            ratio = scales[stage_idx + 1] / scales[stage_idx]
            if config.sigma_policy == "canonical":
                kappa, new_q = aligned_sigma(rsp_q, ratio)
            else:
                kappa, new_q = 1.0, rsp_q
            # Patch the boundary coordinate in the working schedule with the
            # aligned sigma (upstream: `scheduler.sigmas[end] = t_tilde`).
            working_sigmas[global_end] = new_q

            # DCT-expand the video (coupled or fresh band) and rescale by kappa.
            # Coupled policy: the SAME full-grid noise field projected onto
            # each stage's resolution — in the spectral domain. Pixel-space
            # cropping would change the DCT spectrum, so take the combined
            # temporal+spatial DCT of the ORIGINAL full-res noise once and
            # keep only the low-frequency coefficient block matching the
            # next stage's (t, h, w). (With a full-length temporal block this
            # reduces exactly to the spatial-only projection.)
            next_h, next_w, next_t = stage_hw_t[stage_idx + 1]
            if config.noise_policy == "coupled_full_grid":
                full_noise_video, _ = unpack_latent(full_noise)
                full_noise_video = full_noise_video.to(
                    device=internal_video.device, dtype=internal_video.dtype,
                )
                source_t, source_h, source_w = internal_video.shape[-3:]
                full_dct = dct2(dct_temporal(full_noise_video))
                target_dct = full_dct[..., :next_t, :next_h, :next_w] * float(rsp_q)
                target_dct[..., :source_t, :source_h, :source_w] = dct2(
                    dct_temporal(internal_video)
                )
                expanded_video = idct_temporal(idct2(target_dct)).to(
                    dtype=internal_video.dtype
                )
            elif next_t > internal_video.shape[-3]:
                # Temporal expansion needed: use the 3D spectral path.
                expanded_video = spectral_expand_3d(
                    internal_video,
                    (next_t, next_h, next_w),
                    rsp_q,
                    int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                )
            else:
                expanded_video = spectral_expand(
                    internal_video,
                    (next_h, next_w),
                    rsp_q,
                    int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                )
            transitioned_video = expanded_video * kappa

            # Audio handling at this boundary.
            old_audio_sigma = time_shift_sigma(rsp_q, video_shift, audio_shift)
            new_audio_sigma = time_shift_sigma(new_q, video_shift, audio_shift)
            if config.audio_policy == "carry_preserve":
                transitioned_audio = carry_preserved_audio(
                    internal_audio, rsp_q, new_q, old_audio_sigma, new_audio_sigma
                )
            elif config.audio_policy == "clock_reindex":
                if "x0" not in x0_output:
                    raise RuntimeError("clock_reindex requires an x0 callback from this stage")
                _, clean_audio = unpack_latent(x0_output["x0"])
                transitioned_audio = clock_reindex_audio_state(
                    internal_audio,
                    clean_audio,
                    rsp_q,
                    new_q,
                    old_audio_sigma,
                    new_audio_sigma,
                    audio_scale,
                )
            else:
                transitioned_audio = internal_audio

            # Sampler transition hook: once per configured SPEED transition,
            # after the boundary sigma is aligned and patched into the working
            # schedule and the spectral + audio transitions are done, before
            # the next stage re-enters guider.sample(). Stateless samplers
            # no-op here; stateful samplers (PR B) preserve their step
            # history across the boundary. Never called per denoising step
            # and never after the final stage. The per-stream shapes let a
            # stateful handle slice its flat packed history (the real host
            # packs nested latents before the sampler sees them) back into
            # video and audio.
            sampler_handle.on_transition(
                SpeedTransition(
                    stage_idx=stage_idx,
                    ratio=ratio,
                    old_sigma=rsp_q,
                    new_sigma=new_q,
                    source_thw=tuple(internal_video.shape[-3:]),
                    target_thw=(next_t, next_h, next_w),
                    source_stream_shapes=(
                        tuple(public_video.shape),
                        tuple(public_audio.shape),
                    ),
                )
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

            # Advance to the next stage. The next stage's sigma slice derives
            # from GLOBAL boundaries of the working schedule; the boundary
            # coordinate was just patched in place with the aligned sigma.
            stage_start_pub = next_noise
            stage_start_latent = next_zero
            global_start = global_end

        # After the final transition, run the last full-res stage over the
        # remaining working-schedule tail, entering at the aligned boundary sigma
        # (patched into working_sigmas by the last transition).
        log.info("[SPEED] final stage: latent=%s sigmas=%d",
                 list(stage_start_latent.shape) if hasattr(stage_start_latent, 'shape') else stage_start_latent,
                 len(sigmas) - global_start)
        # Final stage is at scale 1.0 (stage n_stages-1) so target == full res.
        # Restore the pristine full-res keyframe/ref latents in the original conds
        # (kept since our first downscale) so the final stage runs exactly like
        # the normal full-res I2V path.
        walker.apply_final()
        final_preview_cb = _wrap_preview_callback(
            stock_cb, x0_output, global_done, global_total,
        )
        final_sigmas = working_sigmas[global_start:]
        final_callback = final_preview_cb
        final_public = guider.sample(
            stage_start_pub,
            stage_start_latent,
            sampler_handle.sampler,
            final_sigmas,
            callback=final_callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        last_public = final_public

        if output_device is not None and last_public is not None:
            last_public = last_public.to(output_device)
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = last_public

        denoised = out
        # The shared x0_output dict holds the most recent denoised x0 (written
        # by the stock callback or the fallback wrapper each step). It is the
        # FULL nested video+audio latent — process_latent_out maps it through
        # whole, so both output LATENTs keep valid video AND audio streams.
        x0 = x0_output.get("x0", None)
        if x0 is not None:
            denoised = latent.copy()
            denoised["samples"] = guider.model_patcher.model.process_latent_out(
                x0.cpu() if hasattr(x0, "cpu") else x0
            )
        return out, denoised
    finally:
        # Run-scoped I2V lifecycle cleanup: on EVERY exit path (including any
        # failure in a stage, transition, audio handling, spectral expansion,
        # or the final sample), restore pristine conditioning latents and
        # remove the walker from the guider. On success apply_final() has
        # already restored + released every wrapper, so this is a no-op.
        # The sampler handle closes first; its failure must not prevent
        # walker cleanup, and a failed walker restore must not prevent
        # dropping the walker from the guider.
        try:
            sampler_handle.close()
        finally:
            try:
                walker.apply_final()
            finally:
                _drop_walker(guider)
