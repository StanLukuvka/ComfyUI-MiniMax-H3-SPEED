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
``ResMultistepState`` that the SPEED runtime can carry across stages
according to the reset-only boundary policy, and reads ``t_prev`` from the
state instead of the schedule.

the historical project-and-rebase comparison path. Experimental V3.0
first-order AUDIO candidate. Temporal transitions clear history instead of
using that operator. Flat host tensors fail closed when recorded stream shape
metadata is absent or inconsistent. These semantics are covered by automated
tests only; this module does not claim GPU or native ComfyUI validation.
"""


from __future__ import annotations

import math
from dataclasses import dataclass

import torch



def _host_ksampler_class():
    """The host's real ``KSAMPLER`` class, resolved lazily.

    Importing it at module load would break ComfyUI-free environments (the
    surrounding package and its tests must import without a full ComfyUI
    install, which only provides ``comfy.samplers`` at runtime). ``None``
    means no real host is installed; then only the sampler-function contract
    is available and the object must not pretend otherwise.
    """
    try:
        from comfy.samplers import KSAMPLER
    except Exception:
        return None
    return KSAMPLER


@dataclass
class ResMultistepState:
    """History maintained by one RES sampler run; the run-scoped handle owns
    the policy for what survives a SPEED boundary.

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
        """Release every previous-step history reference."""
        self.old_denoised = None
        self.old_sigma_down = None
        self.prev_sigma_in = None


def _res_first_order_update(x, denoised, sigma, sigma_down):
    """Return the existing first-order RES candidate for one interval."""
    d = (x - denoised) / sigma
    return x + d * (sigma_down - sigma)


def _res_second_order_update(
    x,
    denoised,
    old_denoised,
    sigma_f,
    sigma_down_f,
    old_sigma_down,
    prev_sigma_in,
):
    """Return the existing second-order RES candidate for one interval."""
    t_old = -math.log(old_sigma_down)
    t_next = -math.log(sigma_down_f)
    t_prev = -math.log(prev_sigma_in)
    h = t_next + math.log(sigma_f)
    c2 = (t_prev - t_old) / h
    phi1 = math.expm1(-h) / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    b1 = 0.0 if math.isnan(phi1 - phi2 / c2) else phi1 - phi2 / c2
    b2 = 0.0 if math.isnan(phi2 / c2) else phi2 / c2
    return math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)


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

    Matches the host ``KSAMPLER`` sampler-FUNCTION contract: called as
    ``fn(model, noise, sigmas, extra_args=..., callback=..., disable=...)``
    where ``model(x, sigma, **extra_args)`` returns the denoised estimate
    and ``callback`` receives the native per-step dict. ``disable`` is
    accepted for contract compatibility; this implementation has no progress
    bar of its own. (The sampler-OBJECT contract is ``.sample`` on
    :class:`ResMultistepSampler`; the two are not interchangeable.)
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
            x_first = _res_first_order_update(x, denoised, sigma, sigma_down)
            x_second = _res_second_order_update(
                x, denoised, old_denoised,
                sigma_f, sigma_down_f, old_sigma_down, prev_sigma_in,
            )
            x = x_second
        else:
            # First order (Euler).
            x = _res_first_order_update(x, denoised, sigma, sigma_down)

        state.old_denoised = denoised
        state.old_sigma_down = sigma_down_f
        state.prev_sigma_in = sigma_f
    return x


class ResMultistepSampler:
    """Host sampler object wrapping the stateful RES function.

    The host guider invokes a sampler *object* as
    ``sampler.sample(guider, sigmas, extra_args, callback, noise,
    latent_image, denoise_mask, disable_pbar)`` (``CFGGuider.inner_sample``
    wraps ``sampler.sample`` in its wrapper executor), which is the contract
    native ``KSAMPLER`` objects implement. This class therefore wraps the
    host's own ``KSAMPLER`` — constructed with :func:`res_multistep_sampler`
    as its sampler function — and delegates ``.sample`` to it, so
    ``noise_scaling``, the inpaint model wrapper, the per-step callback
    adaptation, and ``inverse_noise_scaling`` all stay host-native (plan §13:
    do not bypass ``guider.sample()``).

    ``__call__`` is the separate sampler-*function* contract: it runs
    :func:`res_multistep_sampler` directly on already-host-processed inputs,
    as a plain ``KSAMPLER`` sampler function would receive them. The two
    contracts are not interchangeable; only ``.sample`` satisfies the host
    sampler-object seam.
    """

    def __init__(self, state: ResMultistepState | None = None):
        self.state = state if state is not None else ResMultistepState()
        ksampler_cls = _host_ksampler_class()
        if ksampler_cls is None:
            raise RuntimeError(
                "ResMultistepSampler requires the host comfy.samplers.KSAMPLER; "
                "no real ComfyUI host is available in this environment"
            )
        self._ksampler = ksampler_cls(self._run)

    def _run(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        return res_multistep_sampler(
            model, noise, sigmas, self.state,
            extra_args=extra_args, callback=callback, disable=disable,
        )

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        return self._ksampler.sample(
            model_wrap, sigmas, extra_args, callback, noise,
            latent_image=latent_image, denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )

    def __call__(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        return self._run(
            model, noise, sigmas,
            extra_args=extra_args, callback=callback, disable=disable,
        )
