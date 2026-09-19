"""Automatic MiniMax-H3 SPEED sampler."""

from __future__ import annotations

import comfy.utils

from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.planning import PRESET_TO_STAGES, build_automatic_speed_config
from speed_scripts.sampler_support import SUPPORTED_SPEED_SAMPLERS


class MiniMaxH3SPEEDSampler:
    """Progressive-resolution SPEED sampling with automatic stage placement."""

    DESCRIPTION = (
        "Automatic SPEED sampler — pick stages (2, 3, or 4) and go. "
        "Starts cheap at low resolution, then upsamples when the detail "
        "matters. Set Tolerance (0.5% = 0.005) to trade blur for speed. "
        "Uses baked Euler-derived A/beta; re-calibrate with the Harvest node "
        "if you change checkpoint, LoRA/addons, sampler, or another factor "
        "that materially changes the denoising trajectory."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "stages": ("INT", {"default": 3, "min": 2, "max": 4}),
                "noise_policy": (
                    ["direct_coarse", "coupled_full_grid"],
                    {"default": "direct_coarse"},
                ),
                "Tolerance (Delta)": (
                    "FLOAT",
                    {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001},
                ),
                "noise_amplitude": (
                    "FLOAT",
                    {"default": 12.105, "min": 0.0, "max": 1e6, "step": 0.0001, "round": 0.0001},
                ),
                "noise_decay_exponent": (
                    "FLOAT",
                    {"default": 0.773, "min": 0.0, "max": 10.0, "step": 0.0001, "round": 0.0001},
                ),
                "seed_offset": (
                    "INT",
                    {"default": 10000, "min": 0, "max": 2**31 - 1},
                ),
                "sampler_name": (
                    list(SUPPORTED_SPEED_SAMPLERS),
                    {"default": "euler"},
                ),
            },
        }

    def sample(
        self,
        noise,
        guider,
        sigmas,
        latent_image,
        stages=3,
        noise_policy="direct_coarse",
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
        sampler_name="euler",
        **kwargs,
    ):
        delta = kwargs.get(
            "Tolerance (Delta)",
            kwargs.get(
                "Tolerance",
                kwargs.get("tolerance", kwargs.get("delta", kwargs.get("Delta", 0.005))),
            ),
        )
        if "preset" in kwargs:
            stages = PRESET_TO_STAGES.get(kwargs.pop("preset"), stages)
        try:
            stages = int(stages)
        except Exception:
            stages = 3
        stages = max(2, min(4, stages))

        config = build_automatic_speed_config(
            latent_image,
            stages=stages,
            noise_policy=noise_policy,
            delta=delta,
            noise_amplitude=noise_amplitude,
            noise_decay_exponent=noise_decay_exponent,
            seed_offset=seed_offset,
        )
        return run_speed_pipeline(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            sampler_name=sampler_name,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3SPEEDSampler",
]
