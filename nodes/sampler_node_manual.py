"""Manual step-through SPEED sampler — explicit transition schedule.

Exposes up to four (transition_goal, transition_resolution) pairs.

- goal/resolution == 0 disables that stage.
- steps mode: goal is the global sigma-schedule step index where the stage ends.
- ratio mode: goal is a 0-1 schedule fraction used only to place the boundary.
- resolution is always the stage scale.
- the final active stage's goal is unused; it runs to the end of the schedule.
"""

from __future__ import annotations

import comfy.samplers
import comfy.utils

from speed_scripts.config import RATIO_MODES, SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.nodes_common import full_res_dims, validate_transition_steps


def CALCULATE_SCALES(transitions, ratio_mode):
    """Return active stage scales; goal affects boundary placement, never scale."""
    if ratio_mode not in RATIO_MODES:
        raise ValueError(f"unsupported ratio_mode: {ratio_mode!r}")

    scales = [
        resolution
        for goal, resolution in transitions
        if goal != 0 and resolution != 0
    ]
    if not scales:
        raise ValueError("No valid scales calculated. Check transition goals and resolutions.")
    return scales


class MiniMaxH3SPEEDSamplerManual:
    """SPEED progressive-resolution diffusion with an explicit stage schedule."""

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
        **kwargs,
    ):
        transitions = [
            (float(transition_goal_1), float(transition_resolution_1)),
            (float(transition_goal_2), float(transition_resolution_2)),
            (float(transition_goal_3), float(transition_resolution_3)),
            (float(transition_goal_4), float(transition_resolution_4)),
        ]

        scales = CALCULATE_SCALES(transitions, ratio_mode)
        n_stages = len(scales)
        if n_stages < 2:
            raise ValueError(
                "manual schedule needs at least two active stages (goal != 0) "
                "with the final stage at resolution 1.0"
            )

        total_steps = len(sigmas) - 1
        goals = [goal for goal, resolution in transitions if goal > 0 and resolution != 0]
        transition_goals = goals[:-1]

        if ratio_mode == "steps":
            for goal in transition_goals:
                if goal != int(goal):
                    raise ValueError(
                        f"transition_goal must be a whole step index in steps mode: got {goal}. "
                        "Use ratio_mode='ratio' for fractional (0-1) goals."
                    )
            step_goals = [int(goal) for goal in transition_goals]
        elif ratio_mode == "ratio":
            if any(goal > 1 for goal in transition_goals):
                raise ValueError(
                    f"Invalid goal for ratio mode: {transition_goals}. Goals must be <= 1 "
                    "(fraction of the schedule)."
                )
            step_goals = [int(round(goal * total_steps)) for goal in transition_goals]
        else:
            raise ValueError(f"unsupported ratio_mode: {ratio_mode!r}")

        transition_steps = tuple(step_goals)
        validate_transition_steps(transition_steps, len(sigmas))

        full_h, full_w = full_res_dims(latent_image)
        config = SpeedConfig(
            scales=tuple(scales),
            transition_steps=transition_steps,
            transition_mode="explicit",
            noise_policy=noise_policy,
            delta=0.01,
            transition_seed_offset=int(seed_offset),
            full_latent_h=full_h,
            full_latent_w=full_w,
        )

        # The runtime owns the walker lifecycle: it creates (or reuses) the
        # per-run walker, applies every stage, restores full res, and drops
        # it — including on failure (exception-safe).
        return run_speed_pipeline(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            sampler=comfy.samplers.sampler_object("euler"),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
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
