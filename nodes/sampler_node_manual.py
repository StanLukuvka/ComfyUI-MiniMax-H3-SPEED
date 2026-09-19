"""Manual MiniMax-H3 SPEED sampler with an explicit stage schedule."""

from __future__ import annotations

import comfy.utils

from speed_scripts.config import RATIO_MODES
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.planning import build_manual_speed_config
from speed_scripts.sampler_support import SUPPORTED_SPEED_SAMPLERS


class MiniMaxH3SPEEDSamplerManual:
    """Progressive-resolution SPEED sampling with user-defined stage boundaries."""

    DESCRIPTION = (
        "Manual SPEED sampler — you set the stages by hand. Give up to four "
        "(goal, resolution) pairs: goal = step where that stage ends, resolution = "
        "scale (0.25 = quarter). Set goal or resolution to 0 to skip that stage. "
        "Use this to copy exact paper schedules or to test custom ladders."
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
                "noise_policy": (
                    ["direct_coarse", "coupled_full_grid"],
                    {"default": "direct_coarse"},
                ),
                "seed_offset": (
                    "INT",
                    {"default": 10000, "min": 0, "max": 2**31 - 1},
                ),
                "ratio_mode": (list(RATIO_MODES), {"default": "steps"}),
                "transition_goal_1": ("FLOAT", {"default": 3, "min": 0, "max": 1000}),
                "transition_resolution_1": (
                    "FLOAT",
                    {"default": 0.25, "min": 0, "max": 1},
                ),
                "transition_goal_2": ("FLOAT", {"default": 5, "min": 0, "max": 1000}),
                "transition_resolution_2": (
                    "FLOAT",
                    {"default": 0.5, "min": 0, "max": 1},
                ),
                "transition_goal_3": ("FLOAT", {"default": 8, "min": 0, "max": 1000}),
                "transition_resolution_3": (
                    "FLOAT",
                    {"default": 0.75, "min": 0, "max": 1},
                ),
                "transition_goal_4": ("FLOAT", {"default": 15, "min": 0, "max": 1000}),
                "transition_resolution_4": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 1},
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
        noise_policy="direct_coarse",
        seed_offset=10000,
        ratio_mode="steps",
        transition_goal_1=3,
        transition_resolution_1=0.25,
        transition_goal_2=5,
        transition_resolution_2=0.5,
        transition_goal_3=8,
        transition_resolution_3=0.75,
        transition_goal_4=15,
        transition_resolution_4=1.0,
        sampler_name="euler",
        **kwargs,
    ):
        config = build_manual_speed_config(
            sigmas,
            transitions=(
                (transition_goal_1, transition_resolution_1),
                (transition_goal_2, transition_resolution_2),
                (transition_goal_3, transition_resolution_3),
                (transition_goal_4, transition_resolution_4),
            ),
            ratio_mode=ratio_mode,
            noise_policy=noise_policy,
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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerManual": MiniMaxH3SPEEDSamplerManual}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerManual": "MiniMax H3 SPEED — Sampler (Manual Step-Through)"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3SPEEDSamplerManual",
]
