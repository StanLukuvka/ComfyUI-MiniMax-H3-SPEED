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

# Distinct-boundary latent wrappers — one Latent per keyframe id (and one
# RefLatent per ref2va block). Registry is tiny and lives only for one
# generation; entries are popped at the final-stage restore.
try:
    from .latent import Latent, RefLatent  # type: ignore[import]
except Exception:  # pragma: no cover - import guard for stub tests
    Latent = RefLatent = None  # type: ignore[assignment,misc]

_LATENT_STORE: dict[int, object] = {}

# Pristine condition-latent snapshots. Holders are always plain keyframe dicts
# (the walk in _rescale_cond_latents skips non-dicts), and plain dicts refuse
# setattr, so the id()-keyed module dict is the ONLY store. Entries are popped
# at the final-stage restore so the full-res clones die with each generation.
# Kept in sync with _LATENT_STORE for backwards compat — tests import it.
_PRISTINE_STORE: dict[int, object] = {}

def _get_pristine(holder):
    obj = _LATENT_STORE.get(id(holder))
    if obj is not None and hasattr(obj, "pristine"):
        return obj.pristine
    return _PRISTINE_STORE.get(id(holder))

def _set_pristine(holder, value):
    _PRISTINE_STORE[id(holder)] = value
    obj = _LATENT_STORE.get(id(holder))
    if obj is not None and hasattr(obj, "pristine"):
        try:
            obj.pristine = value  # type: ignore[attr-defined]
        except Exception:
            pass

def _interp_cond(z, h, w):
    """Per-frame bilinear spatial resize of cond latents ([B,C,H,W] or [B,C,T,H,W]).

    torch.nn.functional.interpolate is 4D-native (size=(h,w) on a 5D tensor
    raises), so fold the temporal axis into the batch for the resize and fold
    it back out after — frames are interpolated independently.
    """
    if z.ndim == 4:
        b_, c_, hh, ww = z.shape
        if hh == h and ww == w:
            return z
        resized = torch.nn.functional.interpolate(
            z, size=(h, w), mode="bilinear", align_corners=False,
        )
        return resized.to(dtype=z.dtype)
    if z.ndim != 5:
        raise ValueError(f"unsupported cond latent ndim {z.ndim} (want 4 or 5)")
    b_, c_, t_, hh, ww = z.shape
    if hh == h and ww == w:
        return z
    flattened = z.transpose(1, 2).reshape(b_ * t_, c_, hh, ww)
    resized = torch.nn.functional.interpolate(
        flattened, size=(h, w), mode="bilinear", align_corners=False,
    )
    return resized.reshape(b_, t_, c_, h, w).transpose(1, 2).to(dtype=z.dtype)

def _patch_guider_conditioning_for_stage(guider, stage_h, stage_w, is_final_stage=False):
    """[Level 3] Per-stage I2V fix: rescale the *stored* condition sources.

    Normal ComfyUI I2V path (rev 5749 / c2bcbecd):

    1. ``MiniMaxH3ImageToVideo``(fl2va) / ``MiniMaxH3ReferenceToVideo``(ref2va)
       encode reference frames and stash them on the conditioning via
       ``conditioning_set_values`` under ``minimax_keyframes`` /
       ``minimax_refs``. Each keyframe is ``{"resolved_frame_index": i,
       "latent": z}``; each ref block is ``{"kind": ..., "latent": z, ...}``.
    2. ``CFGGuider.inner_set_conds`` -> ``sampler_helpers.convert_cond`` stores
       that (plus uuid) in ``guider.original_conds["positive"]``. The latents
       are FULL-resolution [B,C,T,H,W] tenors.
    3. On every stage's ``guider.sample()``, ``inner_sample`` copies
       ``original_conds`` into ``self.conds`` and ``process_conds`` calls
       ``model.extra_conds``, which builds ``minimax_payload``::
           payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
       (model_base.py ~2168-2175)
    4. In ``MiniMaxH3Model._forward`` the live latent's coarse dims drive
       ``PackedLayout`` (``frame_rows`` per keyframe), then
       ``all_video_rows[~img_update] = self._cond_video_rows(payload, device)``
       scatters the condition rows. The layout also gets recreated if the
       signature changed; anything that keeps the condition latents full-res
       while the grid shrinks makes ``cond_video_rows`` 4x too long -> the
       [520,96] vs [130,96] broadcast error.

    So the ONLY inputs under our control that end up in ``cond_video_latents``
    are the stored keyframe/ref ``"latent"`` tensors, still full-res inside
    ``original_conds``. (Two traps we fell into previously: the payload dict
    is rebuilt from *these* sources on every guider.sample(), so mutating it
    is a no-op; and ``_cond_video_rows``/PackedLayout live on the
    inner diffusion model, not on the wrapper at ``guider.model_patcher.model``.)

    Fix: walk the positive AND negative conds (negative is kept for parity
    with ComfyUI's own cond handling — MiniMax has no negative prompts in
    practice), find the keyframes / refs, and rescale each
    stored ``latent`` to the stage's (h, w) — rounded to the model's 2x2
    patch multiple is unnecessary because keyframes share the *target* grid
    (PackedLayout computes it), but resize happens from a pristine full-res
    snapshot so progressive stages never degrade the source.

    Args:
        guider: The guider object holding original_conds.
        stage_h: Target height for this stage. If None, restore pristine.
        stage_w: Target width for this stage. If None, restore pristine.
        is_final_stage: If True, restore pristine (full-res) instead of downscaling.
    """
    original_conds = getattr(guider, 'original_conds', None)
    if not isinstance(original_conds, dict):
        return
    for key in ("positive", "negative"):
        conds = original_conds.get(key)
        if not isinstance(conds, list):
            continue
        for cond in conds:
            if not isinstance(cond, dict):
                continue
            _rescale_cond_latents(cond, stage_h, stage_w, is_final_stage=is_final_stage)

def _rescale_cond_latents(cond, stage_h, stage_w, is_final_stage=False):
    """Rescale the fl2va keyframe latents found in one conditioning dict.

    This is the ONLY cond source that can produce the row-mismatch:
    PackedLayout allocates one "cond" segment per keyframe using the LIVE
    target grid (the coarse stage dims), while `_cond_video_rows` patchifies
    the keyframe's own tensor. Full-res tensor + coarse grid = 4x too many
    rows (in the reported crash: 520 full-res vs 130 stage-0 rows).

    ref2va "minimax_refs" blocks are handled by RefLatent (intentionally NOT
    rescaled: their rows are allocated from their own stored latent_h/latent_w
    metadata, which SPEED never changes — tensor and allocation stay consistent
    at every stage, at the cost of running ref rows at full res during coarse
    stages). Keyframe path uses Latent with distinct boundaries:
    input -> scale_to (downsample) -> restore/upscale_to_inject -> release.

    Each keyframe keeps a pristine full-res snapshot (id-keyed module dict,
    popped at the final-stage restore) so the final stage restores full
    resolution and the snapshot dies with its generation.

    Target dims are rounded UP to even (the DiT's 2x2 patch grid): SPEED
    coarse stages can be odd (e.g. 0.25 scale -> w=13), and `patchify_video`
    reshape would crash on odd dims while the live video gets circular-padded
    to even by `pad_to_patch_size` — matching that grid keeps cond rows equal
    to the layout's frame_rows.
    """
    # --- keyframes (fl2va) — rescaled via Latent ---
    keyframes = cond.get("minimax_keyframes", []) or []
    th = stage_h + (stage_h % 2)
    tw = stage_w + (stage_w % 2)
    for kf in keyframes:
        if not isinstance(kf, dict):
            continue
        z = kf.get("latent")
        if z is None or not hasattr(z, "shape"):
            continue
        # Get or create Latent wrapper (distinct input boundary).
        latent_obj = _LATENT_STORE.get(id(kf))
        if latent_obj is None and Latent is not None:
            try:
                latent_obj = Latent(kf)
            except Exception:
                latent_obj = None
            if latent_obj is not None:
                _LATENT_STORE[id(kf)] = latent_obj
                _PRISTINE_STORE[id(kf)] = latent_obj.pristine
        if is_final_stage:
            # Upsample/inject boundary — restore pristine, then consumed.
            if latent_obj is not None and hasattr(latent_obj, "restore"):
                try:
                    before_shape = getattr(z, "shape", None)
                    latent_obj.restore()
                    after_shape = getattr(kf.get("latent"), "shape", None)
                    if before_shape != after_shape:
                        log.info("[SPEED] final stage — restored keyframe latent %s",
                                 list(after_shape) if hasattr(after_shape, "__iter__") else after_shape)
                    # release + pop
                    try:
                        latent_obj.release()
                    except Exception:
                        pass
                except Exception:
                    # Fallback to legacy path
                    pristine = _get_pristine(kf)
                    if pristine is not None and getattr(z, "shape", None) != getattr(pristine, "shape", None):
                        kf["latent"] = pristine.clone()
                        log.info("[SPEED] final stage — restored keyframe latent %s", list(pristine.shape))
            else:
                # Legacy fallback (no Latent class available)
                pristine = _get_pristine(kf)
                if pristine is not None:
                    if getattr(z, "shape", None) != getattr(pristine, "shape", None):
                        kf["latent"] = pristine.clone()
                        log.info("[SPEED] final stage — restored keyframe latent %s", list(pristine.shape))
            _LATENT_STORE.pop(id(kf), None)
            _PRISTINE_STORE.pop(id(kf), None)
            continue
        # Downsample boundary — scale_to from pristine.
        if latent_obj is not None and hasattr(latent_obj, "scale_to"):
            try:
                before_shape = getattr(kf.get("latent"), "shape", None)
                latent_obj.scale_to(stage_h, stage_w)
                after_shape = getattr(kf.get("latent"), "shape", None)
                if before_shape != after_shape:
                    log.info("[SPEED] stage (%d,%d) — keyframe latent %s -> %s",
                             stage_h, stage_w,
                             list(latent_obj.pristine.shape) if hasattr(latent_obj.pristine, "shape") else "?",
                             list(after_shape) if hasattr(after_shape, "__iter__") else after_shape)
            except RuntimeError:
                # already injected/consumed — ignore
                pass
            except Exception:
                # Fallback to legacy interpolate
                pristine = _get_pristine(kf)
                if pristine is not None:
                    src_h, src_w = pristine.shape[-2], pristine.shape[-1]
                    if src_h != th or src_w != tw:
                        kf["latent"] = _interp_cond(pristine, th, tw)
                        log.info("[SPEED] stage (%d,%d) — keyframe latent %s -> %s",
                                 stage_h, stage_w, list(pristine.shape), list(kf["latent"].shape))
        else:
            # Legacy fallback: direct _interp_cond from pristine.
            pristine = _get_pristine(kf)
            if pristine is None:
                try:
                    pristine = z.clone()
                except Exception:
                    continue
                _set_pristine(kf, pristine)
            src_h, src_w = pristine.shape[-2], pristine.shape[-1]
            if src_h != th or src_w != tw:
                kf["latent"] = _interp_cond(pristine, th, tw)
                log.info("[SPEED] stage (%d,%d) — keyframe latent %s -> %s",
                         stage_h, stage_w, list(pristine.shape), list(kf["latent"].shape))

    # --- refs (ref2va) — never rescaled, but lifecycle-tracked via RefLatent ---
    refs = cond.get("minimax_refs", []) or []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        z = ref.get("latent")
        if z is None or not hasattr(z, "shape"):
            continue
        ref_obj = _LATENT_STORE.get(id(ref))
        if ref_obj is None and RefLatent is not None:
            try:
                ref_obj = RefLatent(ref)
            except Exception:
                ref_obj = None
            if ref_obj is not None:
                _LATENT_STORE[id(ref)] = ref_obj
                _PRISTINE_STORE[id(ref)] = ref_obj.pristine
        if is_final_stage:
            if ref_obj is not None and hasattr(ref_obj, "restore"):
                try:
                    ref_obj.restore()
                    try:
                        ref_obj.release()
                    except Exception:
                        pass
                except Exception:
                    pass
            _LATENT_STORE.pop(id(ref), None)
            _PRISTINE_STORE.pop(id(ref), None)
            continue
        if ref_obj is not None and hasattr(ref_obj, "scale_to"):
            try:
                ref_obj.scale_to(stage_h, stage_w)
            except RuntimeError:
                pass
            except Exception:
                pass

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
    # One resolution source: stage_resolution owns this math (the cond-patching
    # path also calls it, so a second inline copy here could silently diverge —
    # exactly the failure mode this pack has a history with).
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
    for ts in transition_steps:
        if not 0 < ts < len(sigmas) - 1:
            raise ValueError("transition step must be inside the sigma schedule")

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

        # I2V per-stage fix: rescale the STORED keyframe/ref latents in
        # guider.original_conds so the per-stage guider.sample() ->
        # process_conds -> model.extra_conds rebuild of minimax_payload
        # picks up cond_video_latents matching this stage's coarse latent.
        # (The payload dict from a previous stage is rebuilt from these
        # sources every call, so these sources are the only patch point.)
        sh, sw, _ = stage_resolution(config, stage_idx, full_h, full_w, full_t)
        _patch_guider_conditioning_for_stage(guider, sh, sw, is_final_stage=False)

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
        q = float(current_sigmas[boundary])
        public_video, public_audio = unpack_latent(public)
        log.info("[SPEED] stage %d output: video=%s audio=%s q=%.4f",
                 stage_idx, list(public_video.shape), list(public_audio.shape), q)

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
                target_dct = dct2(dct_temporal(target_noise)) * float(q)
                target_dct[..., :internal_video.shape[-3], :internal_video.shape[-2], :internal_video.shape[-1]] = source_dct
                expanded_video = idct_temporal(idct2(target_dct))
            else:
                expanded_video = spectral_expand_3d(
                    internal_video,
                    (next_t, next_h, next_w),
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
                (next_h, next_w),
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
    # Final stage is at scale 1.0 (stage n_stages-1) so target == full res.
    # Restore the pristine full-res keyframe/ref latents in the original conds
    # (kept since our first downscale) so the final stage runs exactly like
    # the normal full-res I2V path.
    fh, fw, _ = stage_resolution(config, n_stages - 1, full_h, full_w, full_t)
    _patch_guider_conditioning_for_stage(guider, fh, fw, is_final_stage=True)
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
