"""Stateful RES Multistep adapter for the SPEED pipeline.

Deterministic, non-ancestral RES Multistep (``eta=0``): no noise injection,
no SDE, no CFG++ path. Original implementation written from the published
RES second-order multistep formulation (exponential integrators in
``t = -ln(sigma)`` space). ComfyUI is used only as the behavioral reference
for the host sampler-function contract — its solver source is never copied.

Why this exists: the native ``res_multistep`` sampler keeps its step history
in function locals, so every SPEED stage call would restart with empty
history. Worse, the native code derives the previous input sigma from
``sigmas[i - 1]``, which is the wrong element on the first step of a
stage-local schedule. This adapter owns that history explicitly in a
``ResMultistepState`` that the SPEED runtime carries across stages, and
reads ``t_prev`` from the state instead of the schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .flow import aligned_sigma
from .spectral import spectral_expand_clean_3d


@dataclass
class ResMultistepState:
    """Step history one RES run carries across SPEED stage boundaries.

    ``old_denoised`` is the previous model denoised estimate, ``old_sigma_down``
    the previous interval's destination sigma, and ``prev_sigma_in`` the
    previous interval's input sigma. All three are written after every
    completed interval; a single-sigma stage executes zero intervals and
    leaves them untouched.
    """

    old_denoised: object | None = None
    old_sigma_down: float | None = None
    prev_sigma_in: float | None = None

    def clear(self) -> None:
        """Release every history reference (end-of-run cleanup)."""
        self.old_denoised = None
        self.old_sigma_down = None
        self.prev_sigma_in = None


def project_clean_history(history, target_thw: tuple[int, int, int]):
    """Project one clean RES history to the target video geometry.

    ``history`` is the nested H3 denoised estimate (video ``[B,C,T,H,W]`` +
    audio ``[B,C,2,T_audio]``) stored as solver history in
    ``ResMultistepState.old_denoised``. Only the video geometry is projected,
    with :func:`spectral_expand_clean_3d` — the estimate is clean solver
    history, so it never receives fresh high-frequency noise and is never
    scaled by sigma. The clean audio estimate is preserved unchanged and is
    not sigma-reindexed: the boundary sigma belongs to the noisy re-entry
    state, not to this history.
    """
    if not getattr(history, "is_nested", False):
        raise ValueError("RES clean history must be a nested H3 video/audio pair")
    streams = list(history.unbind())
    if len(streams) != 2:
        raise ValueError("RES clean history must contain exactly video and audio streams")
    video, audio = streams
    projected_video = spectral_expand_clean_3d(video, target_thw)
    return type(history)([projected_video, audio])


def rebase_res_history_sigmas(
    state: ResMultistepState,
    new_sigma: float,
    ratio: float,
) -> None:
    """Rebase the sigma-history fields onto the aligned next-stage coordinates.

    The RES history belongs to the old stage's coordinate system, but the next
    sampler invocation starts at the aligned next-stage boundary. For
    deterministic RES the previous step destination equals the boundary just
    left, so ``old_sigma_down`` becomes ``new_sigma``. ``prev_sigma_in`` maps
    through the same ``aligned_sigma`` scale-coordinate transform the SPEED
    boundary itself uses. Empty fields stay empty. Mutates ``state`` in place;
    never touches the scheduler or the working sigma schedule.
    """
    if state.old_sigma_down is not None:
        state.old_sigma_down = float(new_sigma)
    if state.prev_sigma_in is not None:
        prev = float(state.prev_sigma_in)
        if prev == 1.0:
            # Deviation from the plan's literal "call aligned_sigma": the
            # shared transform rejects q >= 1, but an input sigma of exactly
            # 1.0 is legal history (the first interval of a stage that starts
            # at pure noise). At q = 1 the transform's own formula gives
            # kappa = ratio / ratio = 1, i.e. the identity, so the rebased
            # value stays 1.0. Every other out-of-domain sigma (> 1 or <= 0)
            # still fails closed through aligned_sigma.
            state.prev_sigma_in = prev
        else:
            _, state.prev_sigma_in = aligned_sigma(prev, ratio)


def res_multistep_sampler(
    model,
    noise,
    sigmas,
    state: ResMultistepState,
    extra_args=None,
    callback=None,
    disable=None,
):
    """Run deterministic RES Multistep over ``sigmas``, carrying ``state``.

    Matches the host ``KSAMPLER`` sampler-function contract: called as
    ``fn(model, noise, sigmas, extra_args=..., callback=..., disable=...)``
    where ``model(x, sigma, **extra_args)`` returns the denoised estimate
    and ``callback`` receives the native per-step dict. ``disable`` is
    accepted for contract compatibility; this implementation has no progress
    bar of its own.
    """
    extra_args = {} if extra_args is None else extra_args
    x = noise
    s_in = x.new_ones([x.shape[0]]) if torch.is_tensor(x) else 1.0

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in, **extra_args)
        # Deterministic RES (eta=0): the destination sigma is the next
        # schedule entry and no noise is ever added.
        sigma_down = sigmas[i + 1]
        if callback is not None:
            callback({
                "x": x, "i": i, "sigma": sigma, "sigma_hat": sigma,
                "denoised": denoised,
            })

        sigma_f = float(sigma)
        sigma_down_f = float(sigma_down)
        old_denoised = state.old_denoised
        old_sigma_down = state.old_sigma_down
        prev_sigma_in = state.prev_sigma_in
        # First order when the destination sigma is zero, when the carried
        # history is missing, or when the interval is degenerate (zero width
        # or a zero previous-step gap, which would divide by zero below).
        # For every well-formed strictly decreasing schedule this reduces to
        # the plan rule: first order iff destination sigma is zero or
        # old_denoised is missing.
        use_second_order = (
            0.0 < sigma_down_f < sigma_f
            and old_denoised is not None
            and old_sigma_down is not None
            and prev_sigma_in is not None
            and old_sigma_down != prev_sigma_in
        )
        if use_second_order:
            # Second order RES multistep: exponential integrators in
            # t = -ln(sigma) space. t_prev comes from the carried state,
            # never from sigmas[i - 1]: on the first step of a stage-local
            # schedule that index is the wrong element.
            t_old = -math.log(old_sigma_down)
            t_next = -math.log(sigma_down_f)
            t_prev = -math.log(prev_sigma_in)
            h = t_next + math.log(sigma_f)
            c2 = (t_prev - t_old) / h
            phi1 = math.expm1(-h) / (-h)
            phi2 = (phi1 - 1.0) / (-h)
            b1 = 0.0 if math.isnan(phi1 - phi2 / c2) else phi1 - phi2 / c2
            b2 = 0.0 if math.isnan(phi2 / c2) else phi2 / c2
            x = math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)
        else:
            # First order (Euler).
            d = (x - denoised) / sigma
            x = x + d * (sigma_down - sigma)

        state.old_denoised = denoised
        state.old_sigma_down = sigma_down_f
        state.prev_sigma_in = sigma_f
    return x


class ResMultistepSampler:
    """Host ``KSAMPLER``-compatible sampler object wrapping the stateful
    RES function.

    ``KSAMPLER`` invokes its sampler function as
    ``fn(model, noise, sigmas, extra_args=..., callback=..., disable=...)``;
    this object exposes exactly that call shape and always routes through
    one shared ``ResMultistepState`` so the history survives every SPEED
    stage call within the run.
    """

    def __init__(self, state: ResMultistepState | None = None):
        self.state = state if state is not None else ResMultistepState()

    def __call__(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        return res_multistep_sampler(
            model, noise, sigmas, self.state,
            extra_args=extra_args, callback=callback, disable=disable,
        )
