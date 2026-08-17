"""MiniMaxH3SPEEDSampler — SPEED as a ComfyUI SAMPLER node.

Ported from the Lab's sampler_speed.py. Returns a standard SAMPLER that slots
into ComfyUI's SamplerCustomAdvanced (which owns the guider/CFG conditioning
and calls ``guider.sample``). Inside that flow the K-Sampler's ``sample_*``
function is invoked on a FLAT PACKED ``[B,1,N]`` tensor (video+audio
concatenated along N) with only ``model_options`` and ``seed`` in
``extra_args`` — the per-stream geometry is NOT recoverable from there. It is
delivered via the KSAMPLER's ``extra_options`` as a ``latent_shapes`` kwarg,
populated from the node's optional ``latent`` input.

The SPEED transition math (delta-optimal steps, DCT spectral expand) is reused
from the runtime and spectral helpers; nothing here duplicates them.
"""

from __future__ import annotations

import torch

from minimax_h3_speed.config import (
    SpeedConfig,
    SCALE_PRESETS,
    DEFAULT_TRANSITION_STEPS,
    canonical_config,
)
from minimax_h3_speed.h3_runtime import resolve_transition_steps
from minimax_h3_speed.spectral import spectral_expand_dct


def _recover_streams(x: torch.Tensor, latent_shapes):
    """Split a flat-packed [B,1,N] tensor back into its video/audio streams.

    Returns (video, audio) for the nested H3 case, or (x, None) when
    ``latent_shapes`` is a single stream (or absent) meaning ``x`` is already a
    plain 4D/5D latent.
    """
    if latent_shapes is None or len(latent_shapes) < 2:
        return x, None
    import comfy.utils

    streams = comfy.utils.unpack_latents(x, latent_shapes)
    return streams[0], streams[1]


def _repack_streams(video: torch.Tensor, audio: torch.Tensor):
    """Pack (video, audio) back into a single [B,1,N] tensor."""
    import comfy.utils

    if audio is None:
        return video
    packed, _ = comfy.utils.pack_latents([video, audio])
    return packed


def _unpack_streams(packed: torch.Tensor, shapes) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a flat-packed tensor using a (video, audio) shape list."""
    import comfy.utils

    streams = comfy.utils.unpack_latents(packed, shapes)
    return streams[0], streams[1]


def _segment_callback(outer_cb, segment_start_idx: int):
    """Re-base k-diffusion callback step indices onto the full schedule."""
    if outer_cb is None:
        return None

    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + segment_start_idx
        outer_cb(d)

    return inner


def sample_speed_packed(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    latent_shapes=None,
    config: SpeedConfig | None = None,
) -> torch.Tensor:
    """ComfyUI ``sample_*``-compatible SPEED sampler for H3 packed latents.

    Segments the denoising trajectory at each delta-optimal (or explicit) SPEED
    transition, DCT-expanding the VIDEO spatial axes between segments and
    re-pack[ing] video+audio before each model call.

    ``latent_shapes`` arrives via the KSAMPLER ``extra_options`` (never
    ``extra_args``). When it is a two-stream list the input/output are nested
    packed [B,1,N]; when single/None the tensor is treated as a plain latent.
    """
    import comfy.k_diffusion.sampling as kds

    if config is None:
        config = canonical_config()
    extra_args = {} if extra_args is None else extra_args

    # The base integrator: ordinary Euler (k-diffusion), the same solver the
    # guider-based stage runner uses internally.
    sampler_fn = getattr(kds, "sample_euler")  # stable; no Configurable-style knob here yet

    video, audio = _recover_streams(x, latent_shapes)
    H_full, W_full = video.shape[-2], video.shape[-1]

    # Per-segment pack/unpack shapes. The video's H/W changes across transitions,
    # so we rebuild the pack shape list each segment from the CURRENT video shape
    # (never the original full-res latent_shapes).
    audio_shape = None if audio is None else tuple(audio.shape)

    def current_shapes():
        if audio is None:
            return None
        return [list(video.shape), list(audio_shape)]

    # Build per-preset scale ladder.
    scales = list(config.scales)
    if len(scales) < 2:
        # Not progressive: plain Euler passthrough (flat pack, no unpack).
        return sampler_fn(model, x, sigmas, extra_args=extra_args,
                          callback=callback, disable=disable)

    # DCT-truncate the incoming latent down to the coarsest scale.
    if scales[0] < 1.0:
        video = _initial_dct_downscale(video, scales[0])

    transitions = _resolve_transitions(config, sigmas, scales, H_full=H_full, W_full=W_full)
    sigmas = sigmas.clone()
    segment_starts = [0] + [t[0] for t in transitions]

    for seg_i, seg_start in enumerate(segment_starts):
        seg_end = transitions[seg_i][0] if seg_i < len(transitions) else len(sigmas) - 1
        seg_sigmas = sigmas[seg_start:seg_end + 1]
        shapes = current_shapes()
        if len(seg_sigmas) >= 2:
            x_seg = _repack_streams(video, audio)
            cb = _segment_callback(callback, seg_start)
            x_seg = sampler_fn(model, x_seg, seg_sigmas, extra_args=extra_args,
                               callback=cb, disable=disable)
            if shapes is None:
                video = x_seg       # single-stream: x is the video already
            else:
                video, audio = _unpack_streams(x_seg, shapes)

        if seg_i >= len(transitions):
            break

        step_idx, s_i, s_next = transitions[seg_i]
        sigma_at_transition = float(sigmas[step_idx])
        video, t_tilde = _expand_video_and_align(
            video, s_i, s_next, sigma_at_transition,
            H_full=H_full, W_full=W_full, seed_offset=config.transition_seed_offset, seg_i=seg_i,
        )
        # Patch only the transition sigma in the full schedule. This matches the
        # reference inference loop which edits sigmas[step_idx] = t_tilde.
        sigmas[step_idx] = float(t_tilde)

    return _repack_streams(video, audio)


def _resolve_transitions(config: SpeedConfig, sigmas, scales, *, H_full: int, W_full: int):
    """Return [(step_idx, s_i, s_next)] transitions (may be empty)."""
    if len(scales) < 2:
        return []
    steps = resolve_transition_steps(config, sigmas, H_full, W_full)  # delta-optimal or explicit
    out = []
    n_steps = len(sigmas) - 1
    for i, (s_old, s_next) in enumerate(zip(scales[:-1], scales[1:])):
        step = steps[i] if i < len(steps) else n_steps
        if step >= n_steps:
            break
        out.append((int(step), float(s_old), float(s_next)))
    return out


def _initial_dct_downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """DCT-truncate ``x`` [..., H, W] down to ``scale`` of full resolution."""
    if scale >= 1.0:
        return x
    H_full, W_full = x.shape[-2], x.shape[-1]
    H_lo, W_lo = round(H_full * scale), round(W_full * scale)
    if H_lo < 1 or W_lo < 1:
        raise ValueError("coarsest scale collapses spatial axes")
    target = (H_lo, W_lo)
    from minimax_h3_speed.spectral import lowpass_dct

    return lowpass_dct(x, target)


def _expand_video_and_align(
    video: torch.Tensor,
    s_i: float,
    s_next: float,
    sigma_at_transition: float,
    *,
    H_full: int,
    W_full: int,
    seed_offset: int,
    seg_i: int,
) -> tuple[torch.Tensor, float]:
    """Expand video spatial axes to the next scale, returning (expanded, t_tilde).

    Uses the lab's pure-torch DCT expand (seeded high-freq fill at the transition
    sigma) and the repo's ``aligned_speed_sigma`` for the kappa rescale and the
    aligned next timestep — identical math to the reference's
    ``_expand_and_align_torch`` but on the repo's own spectral primitives.
    """
    from minimax_h3_speed.flow import aligned_speed_sigma

    source_h, source_w = video.shape[-2], video.shape[-1]
    H_tgt = round(s_next * H_full)
    W_tgt = round(s_next * W_full)
    if H_tgt < source_h or W_tgt < source_w:
        raise ValueError(f"DCT cannot expand {(source_h, source_w)} to {(H_tgt, W_tgt)}")

    q = float(sigma_at_transition)
    ratio = s_next / s_i
    kappa, t_tilde = aligned_speed_sigma(q, ratio)

    seed = seed_offset + (seg_i + 1) * 10_000
    expanded = spectral_expand_dct(video, (H_tgt, W_tgt), q, seed)
    expanded = expanded * float(kappa)
    return expanded.to(dtype=video.dtype), float(t_tilde)


class MiniMaxH3SPEEDSampler:
    """SPEED sampler node returning a ComfyUI SAMPLER for SamplerCustomAdvanced."""

    DESCRIPTION = (
        "SPEED progressive-resolution diffusion as a SAMPLER, composable in "
        "ComfyUI's SamplerCustomAdvanced. Takes the H3 latent (optional) only to "
        "read video/audio geometry; returns a SAMPLER."
    )
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "make"
    CATEGORY = "sampling/minimax_h3_speed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(SCALE_PRESETS.keys()),),
                "transition_mode": (["delta_custom", "explicit"],),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "power_A": ("FLOAT", {"default": 219.48, "min": 0.0, "max": 1e6}),
                "power_beta": ("FLOAT", {"default": 2.42, "min": 0.0, "max": 10.0}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "latent": ("LATENT",),  # read video/audio geometry; not a sampling input
            },
        }

    def make(self, preset, transition_mode, delta, power_A, power_beta,
             seed_offset, latent=None):
        import comfy.samplers

        _latent_shapes = None
        if latent is not None:
            nested = latent.get("samples")
            if nested is not None and getattr(nested, "is_nested", False):
                _latent_shapes = [tuple(s.shape) for s in nested.unbind()]
            elif nested is not None:
                _latent_shapes = [tuple(nested.shape)]

        config = SpeedConfig(
            scales=SCALE_PRESETS[preset],
            transition_steps=DEFAULT_TRANSITION_STEPS[preset],
            transition_mode=transition_mode,
            delta=float(delta),
            power_A=float(power_A),
            power_beta=float(power_beta),
            transition_seed_offset=int(seed_offset),
        )
        sampler = comfy.samplers.KSAMPLER(
            sample_speed_packed,
            extra_options={"latent_shapes": _latent_shapes, "config": config},
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler (Canonical)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler", "sample_speed_packed"]
