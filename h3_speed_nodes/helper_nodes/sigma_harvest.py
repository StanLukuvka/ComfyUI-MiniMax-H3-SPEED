"""SigmaHarvest node — runs one Euler pass, captures residual spectrum.

Uses the H3 nested latent geometry. Zeroes the latent values before the pass so
the measured power is from clean noise, not content. Returns JSON for
HarvestToConfig.
"""

from __future__ import annotations

import json

import torch

import comfy.samplers
import comfy.utils

from minimax_h3_speed.harvest import HarvestCallback
from minimax_h3_speed.h3_runtime import _unpack_tensor, _pack_tensor


class MiniMaxH3SigmaHarvest:
    DESCRIPTION = (
        "Runs one Euler pass over a MiniMax-H3 nested latent, harvesting the "
        "residual (x - x0) power spectrum at each sigma level. Fits "
        "P = A·omega^(-beta) and emits a calibration payload (JSON) for "
        "HarvestToConfig / Schedule."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("harvest_json",)
    FUNCTION = "run"
    CATEGORY = "sampling/minimax_h3_speed/diagnostics"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "capture_every": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
                "fit_mode": (["first", "per_sigma", "pooled"],),
                "omega_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
            },
        }

    def run(self, noise, guider, sigmas, latent_image,
            capture_every=1, fit_mode="first", omega_min=0.5, delta=0.01):
        # Validate H3 nested latent
        samples = latent_image.get("samples")
        if not getattr(samples, "is_nested", False):
            raise ValueError("MiniMax-H3 SPEED requires a NestedTensor video/audio latent")
        full_video, full_audio = _unpack_tensor(samples)
        full_h, full_w = full_video.shape[-2:]

        # Zero the latent so measured power is from clean noise (not content).
        zero_latent = latent_image.copy()
        zero_latent["samples"] = _pack_tensor(
            torch.zeros_like(full_video),
            torch.zeros_like(full_audio),
        )

        # Run one Euler pass with our capture callback.
        callback = HarvestCallback(sigmas=sigmas, every=int(capture_every))

        guider.sample(
            noise,
            zero_latent["samples"],
            comfy.samplers.sampler_object("euler"),
            sigmas,
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise.seed,
        )

        # Finalize: fit power law, recommend configs.
        result = callback.finalize(
            omega_min=float(omega_min),
            latent_h=full_h,
            latent_w=full_w,
            delta=float(delta),
            fit_mode=fit_mode,
        )
        return (json.dumps(result, indent=2, default=str),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SigmaHarvest": MiniMaxH3SigmaHarvest}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SigmaHarvest": "MiniMax H3 SPEED — Sigma Harvest"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3SigmaHarvest"]
