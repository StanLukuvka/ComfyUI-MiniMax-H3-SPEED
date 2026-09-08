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

from .latent_class import LatentClass, LatentStage, LatentWalker
from .observer import (
    SpeedRunEndEvent,
    SpeedRunStartEvent,
    SpeedStepEvent,
    SpeedTransitionEvent,
)


# Per-pipeline-run walker, stashed on the guider so the same wrapper dict
# survives every coarse stage and the final restore. Dropped at the end of
# every run; recreated on the next run that touches the same guider.
_LW_ATTR = "_speed_latent_walker"


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

def _wrap_observer_callback(inner_cb, observer, event_fn):
    """Compose the observer around an existing stage callback (plan §14).

    Order per callback: capture x0 / forward stock preview + progress
    (the inner `_wrap_preview_callback` wrapper) FIRST, then notify the
    observer — the UI update must not wait on observer work. The observer
    receives the exact callback x0/x the inner wrapper recorded; the
    shared `x0_output` dict remains the single x0 source (plan §45).
    """
    if observer is None:
        return inner_cb

    def callback(step, x0, x, total_steps):
        inner_cb(step, x0, x, total_steps)
        observer.on_step(event_fn(step), x0, x)

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
    sampler,
    # PR3 (progress-bar): KEEP THE PROGRESS BAR ON BY DEFAULT. disable_pbar
    # defaults to False (bar VISIBLE). The SPEED sampler node explicitly passes
    # `disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED` to honor the user's
    # ComfyUI setting. DO NOT change this default back to True — doing so hides
    # the progress bar for every run and is easy to miss (the node still "works").
    disable_pbar: bool = False,
    output_device=None,
    preview_callback=None,
    x0_output=None,
    observer=None,
):
    """[Level 1] Run an N-stage progressive-resolution Euler chain (multi-stage SPEED).

    This is intentionally the slow correctness oracle: each public guider call
    performs its own prepare/pre-run/cleanup lifecycle and naturally rebuilds
    H3's shape-dependent conditions.

    Mirrors canonical SPEED ``generate``: it computes transition steps from
    delta-optimal thresholds, runs each scale stage, and DCT-expands + kappa-aligns
    at each boundary.

    Live previews are forwarded from ComfyUI's `latent_preview.prepare_callback`
    pipeline (the same one `SamplerCustomAdvanced` uses). Pass a callback built
    by the node layer in `preview_callback`; if `None`, the function builds one
    for the current run via `_build_preview_callback` so previews keep working
    in offline tests and programmatic use. The shared `x0` dict (`x0_output`)
    is also forwarded so `denoised` is reconstructed exactly as in
    `SamplerCustomAdvanced`.

    `observer` optionally receives runtime events (see `speed_scripts.observer`):
    one `on_step` per actual denoising interval, one `on_transition` per
    resolution transition (coincident transitions included — a zero-step
    intermediate stage emits no step event but its transition still fires),
    plus one `on_run_start` / `on_run_end` each. `None` (the default) runs
    the pipeline exactly as before.

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

    if observer is not None:
        observer.on_run_start(SpeedRunStartEvent(
            n_stages=n_stages,
            scales=tuple(scales),
            transition_steps=tuple(int(s) for s in transition_steps),
            global_steps=global_total,
            full_h=full_h,
            full_w=full_w,
            full_t=full_t,
        ))

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
        walker = _get_or_create_walker(guider)
        walker.apply_stage(rsp_sh, rsp_sw)

        # Run the current stage over its global slice
        # (`working_sigmas[global_start : global_end + 1]`). The wrapped
        # callback forwards to the shared stock callback
        # (`latent_preview.prepare_callback`) with the step remapped to the
        # global timeline, so the bar runs continuously and preview bytes
        # flow through the same PROGRESS_BAR_HOOK every other ComfyUI node
        # uses. The wrapper writes x0 into the shared `x0_output` dict for
        # `clock_reindex` audio and the denoised fallback. The observer
        # wrapper composes AROUND that preview wrapper so the stock
        # preview/progress update always runs before observer work.
        preview_cb = _wrap_preview_callback(
            stock_cb, x0_output, global_done, global_total,
        )
        stage_sigmas = working_sigmas[global_start:global_end + 1]
        stage_h, stage_w, stage_t = stage_hw_t[stage_idx]

        def _step_event(stage_local_step, _stage_idx=stage_idx,
                        _global_start=global_start, _stage_sigmas=stage_sigmas,
                        _stage_scale=scales[stage_idx], _stage_h=stage_h,
                        _stage_w=stage_w, _stage_t=stage_t,
                        _callback_offset=global_done):
            _global_index = _global_start + stage_local_step
            return SpeedStepEvent(
                callback_index=_callback_offset + stage_local_step,
                stage_index=_stage_idx,
                stage_scale=_stage_scale,
                stage_local_step=stage_local_step,
                global_schedule_index=_global_index,
                actual_sigma=float(_stage_sigmas[stage_local_step]),
                actual_sigma_next=float(_stage_sigmas[stage_local_step + 1]),
                original_sigma=float(sigmas[_global_index]),
                original_sigma_next=float(sigmas[_global_index + 1]),
                stage_h=_stage_h,
                stage_w=_stage_w,
                stage_t=_stage_t,
                full_h=full_h,
                full_w=full_w,
                full_t=full_t,
            )

        callback = _wrap_observer_callback(preview_cb, observer, _step_event)
        # H3-runtime behavior: a zero-step intermediate stage still invokes
        # guider.sample once with a single-sigma schedule (zero denoising
        # steps). Upstream SPEED skips the sampler for such segments.
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

        if observer is not None:
            observer.on_transition(SpeedTransitionEvent(
                transition_index=stage_idx,
                from_stage=stage_idx,
                to_stage=stage_idx + 1,
                global_schedule_index=global_end,
                from_scale=scales[stage_idx],
                to_scale=scales[stage_idx + 1],
                scale_ratio=ratio,
                sigma_before_alignment=rsp_q,
                sigma_after_alignment=new_q,
                kappa=kappa,
                source_h=stage_hw_t[stage_idx][0],
                source_w=stage_hw_t[stage_idx][1],
                target_h=stage_hw_t[stage_idx + 1][0],
                target_w=stage_hw_t[stage_idx + 1][1],
            ))

        # DCT-expand the video (coupled or fresh band) and rescale by kappa.
        next_h, next_w, next_t = stage_hw_t[stage_idx + 1]
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
                target_dct = dct2(dct_temporal(target_noise)) * float(rsp_q)
                target_dct[..., :internal_video.shape[-3], :internal_video.shape[-2], :internal_video.shape[-1]] = source_dct
                expanded_video = idct_temporal(idct2(target_dct))
            else:
                expanded_video = spectral_expand_3d(
                    internal_video,
                    (next_t, next_h, next_w),
                    rsp_q,
                    int(noise.seed) + int(config.transition_seed_offset) + stage_idx,
                )
        elif config.noise_policy == "coupled_full_grid":
            full_noise_video, _ = unpack_latent(full_noise)
            expanded_video = spectral_expand_coupled(
                internal_video,
                full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype),
                rsp_q,
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
    fh, fw, _ = stage_resolution(config, n_stages - 1, full_h, full_w, full_t)
    walker = _get_or_create_walker(guider)
    walker.apply_final()
    _drop_walker(guider)
    final_preview_cb = _wrap_preview_callback(
        stock_cb, x0_output, global_done, global_total,
    )
    final_sigmas = working_sigmas[global_start:]

    def _final_step_event(stage_local_step, _stage_idx=n_stages - 1,
                          _global_start=global_start, _stage_sigmas=final_sigmas,
                          _stage_scale=scales[n_stages - 1],
                          _stage_h=fh, _stage_w=fw, _stage_t=stage_hw_t[n_stages - 1][2],
                          _callback_offset=global_done):
        _global_index = _global_start + stage_local_step
        return SpeedStepEvent(
            callback_index=_callback_offset + stage_local_step,
            stage_index=_stage_idx,
            stage_scale=_stage_scale,
            stage_local_step=stage_local_step,
            global_schedule_index=_global_index,
            actual_sigma=float(_stage_sigmas[stage_local_step]),
            actual_sigma_next=float(_stage_sigmas[stage_local_step + 1]),
            original_sigma=float(sigmas[_global_index]),
            original_sigma_next=float(sigmas[_global_index + 1]),
            stage_h=_stage_h,
            stage_w=_stage_w,
            stage_t=_stage_t,
            full_h=full_h,
            full_w=full_w,
            full_t=full_t,
        )

    final_callback = _wrap_observer_callback(
        final_preview_cb, observer, _final_step_event,
    )
    final_public = guider.sample(
        stage_start_pub,
        stage_start_latent,
        sampler,
        final_sigmas,
        callback=final_callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    last_public = final_public

    if observer is not None:
        observer.on_run_end(SpeedRunEndEvent(
            n_stages=n_stages,
            global_steps=global_total,
            transition_count=n_stages - 1,
        ))

    if output_device is not None and last_public is not None:
        last_public = last_public.to(output_device)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = last_public

    denoised = out
    # The shared x0_output dict holds the most recent denoised x0 (written
    # by the stock callback or the fallback wrapper each step).
    x0 = x0_output.get("x0", None)
    if x0 is not None:
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
