"""Use deterministic RES Multistep across SPEED stages.

ComfyUI normally keeps RES history inside one sampler call. SPEED uses several
sampler calls, so this adapter stores that history explicitly. History is
cleared whenever SPEED changes resolution, then RES starts building it again.

Only deterministic RES is supported here: no ancestral sampling, SDE, or CFG++.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _host_ksampler_class():
    """Load ComfyUI's ``KSAMPLER`` only when RES is used."""
    try:
        from comfy.samplers import KSAMPLER
    except Exception:
        return None
    return KSAMPLER


@dataclass
class ResMultistepState:
    """The previous-step values RES needs for its next update."""

    old_denoised: object | None = None
    old_sigma_down: float | None = None
    prev_sigma_in: float | None = None

    def clear(self) -> None:
        """Clear RES history before starting at a new resolution."""
        self.old_denoised = None
        self.old_sigma_down = None
        self.prev_sigma_in = None


def _res_first_order_update(x, denoised, sigma, sigma_down):
    """Run one first-order RES step."""
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
    """Run one second-order RES step."""
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
    """Run RES across the sigma schedule while keeping its step history."""
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
    """ComfyUI sampler wrapper for the stateful RES function."""

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
