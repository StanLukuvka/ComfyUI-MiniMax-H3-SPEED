"""SigmaHarvest node — runs one Euler pass, captures residual spectrum.

Flat root-level module, same registration pattern as sampler_node.py:
export NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS and let the root
__init__.py import this file directly (no subpackage).
"""

from __future__ import annotations

import json

import torch

import comfy.samplers
import comfy.utils

from h3_logging import get_logger
from minimax_h3_speed.harvest import HarvestCallback
from minimax_h3_speed.h3_runtime import _unpack_tensor, _pack_tensor

log = get_logger()


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
            log.error("latent is not a NestedTensor (is_nested=%r, type=%s); "
                      "the upstream node must be MiniMaxH3ImageToVideo",
                      getattr(samples, "is_nested", None), type(samples).__name__)
            raise ValueError("MiniMax-H3 SPEED requires a NestedTensor video/audio latent")
        full_video, full_audio = _unpack_tensor(samples)
        full_h, full_w = full_video.shape[-2:]
        log.info("SigmaHarvest: video=%s audio=%s sigma_count=%d capture_every=%d fit=%s",
                 tuple(full_video.shape), tuple(full_audio.shape),
                 int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas),
                 int(capture_every), fit_mode)

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
        log.info("SigmaHarvest done: fit_mode=%s A=%.3f beta=%.3f r2=%.3f health=%s",
                 fit_mode,
                 float(result["overall_fit_A"]),
                 float(result["overall_fit_beta"]),
                 float(result["overall_fit_r2"]),
                 result["fit_health"])
        return (json.dumps(result, indent=2, default=str),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SigmaHarvest": MiniMaxH3SigmaHarvest}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SigmaHarvest": "MiniMax H3 SPEED — Sigma Harvest"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3SigmaHarvest"]
