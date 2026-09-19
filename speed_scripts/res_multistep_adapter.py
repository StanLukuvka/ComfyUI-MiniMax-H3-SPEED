"""Stateful deterministic RES Multistep adapter for the SPEED pipeline.

The native ComfyUI sampler keeps previous-step history in function locals, so
separate ``guider.sample()`` calls do not share it. This adapter makes that
history explicit while preserving the host sampler-object contract. SPEED's
shipping boundary policy is reset-only: the run-scoped handle clears history
at every resolution transition, so the first real interval at the new
resolution cold-starts before normal RES multistep history rebuilds.

Only deterministic, non-ancestral RES is supported here (``eta=0``, no SDE,
no CFG++ path).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _host_ksampler_class():
    """Resolve the host's real ``KSAMPLER`` lazily."""
    try:
        from comfy.samplers import KSAMPLER
    except Exception:
        return None
    return KSAMPLER


@dataclass
class ResMultistepState:
    """Previous-step history owned by one RES sampler run."""

    old_denoised: object | None = None
    old_sigma_down: float | None = None
    prev_sigma_in: float | None = None

    def clear(self) -> None:
        """Release every previous-step history reference."""
        self.old_denoised = None
        self.old_sigma_down = None
        self.prev_sigma_in = None


def _res_first_order_update(x, denoised, sigma, sigma_down):
    """Return the deterministic first-order RES update for one interval."""
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
    """Return the deterministic second-order RES update for one interval."""
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
    """Run deterministic RES Multistep over ``sigmas`` while carrying ``state``."""
    extra_args = {} if extra_args is None else extra_args
    x = noise
    s_in = x.new_ones([x.shape[0]]) if torch.is_tensor(x) else 1.0

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in, **extra_args)
        sigma_down = sigmas[i + 1]
        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sigma_hat": sigma,
                "denoised": denoised,
            })

        sigma_f = float(sigma)
        sigma_down_f = float(sigma_down)
        old_denoised = state.old_denoised
        old_sigma_down = state.old_sigma_down
        prev_sigma_in = state.prev_sigma_in

        use_second_order = (
            0.0 < sigma_down_f < sigma_f
            and old_denoised is not None
            and old_sigma_down is not None
            and prev_sigma_in is not None
            and old_sigma_down != prev_sigma_in
        )
        if use_second_order:
            x = _res_second_order_update(
                x,
                denoised,
                old_denoised,
                sigma_f,
                sigma_down_f,
                old_sigma_down,
                prev_sigma_in,
            )
        else:
            x = _res_first_order_update(x, denoised, sigma, sigma_down)

        state.old_denoised = denoised
        state.old_sigma_down = sigma_down_f
        state.prev_sigma_in = sigma_f
    return x


class ResMultistepSampler:
    """Host sampler object wrapping the stateful deterministic RES function."""

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
            model,
            noise,
            sigmas,
            self.state,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
        )

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        return self._ksampler.sample(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image=latent_image,
            denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )

    def __call__(
        self,
        model,
        noise,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
    ):
        return self._run(
            model,
            noise,
            sigmas,
            extra_args=extra_args,
            callback=callback,
            disable=disable,
        )
