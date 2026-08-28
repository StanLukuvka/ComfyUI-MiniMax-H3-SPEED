"""Manual step-through SPEED sampler — uses LatentClass directly (no build_config_and_run).

Exposes the transition schedule directly: up to four
(transition_goal, transition_resolution) widget pairs.

Semantics:

- ``transition_goal_N == 0`` or ``transition_resolution_N == 0`` disables stage N.
- ``ratio_mode == "steps"`` (default): goal is a STEP INDEX (position in the
  sigma schedule) at which stage N ends; resolution is that stage's scale.
- ``ratio_mode == "ratio"``: goal must be <= 1. Scale is
  ``resolution * goal``; the boundary is placed at ``round(goal * total_steps)``.

The cond-patching is done via LatentClass.prime + the runtime's walk_guider shim.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.config import RATIO_MODES, SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline, unpack_latent
from speed_scripts.latent_class import LatentClass
from speed_scripts.nodes_common import validate_transition_steps


def CALCULATE_SCALES(transitions, ratio_mode):
    """Build the per-stage scale factors from (goal, resolution) pairs.

    Steps mode: resolution is the stage's scale. Ratio mode: the goal is a
    fraction of the schedule, and the scale must be scaled by it. A goal of 0
    skips that stage; later stages stay active and shift down.
    """
    scales = []
    for goal, resolution in transitions:
        if goal == 0 or resolution == 0:
            continue
        if ratio_mode == "steps":
            scales.append(resolution)
        elif ratio_mode == "ratio":
            if goal > 1:
                raise ValueError(f"Invalid goal for ratio mode: {goal}. Goal must be <= 1.")
            scales.append(resolution * goal)
    if not scales:
        raise ValueError("No valid scales calculated. Check transition goals and resolutions.")
    return scales


class MiniMaxH3SPEEDSamplerManualClass:
    """Manual step-through SPEED sampler using LatentClass.

    Same as the original `MiniMaxH3SPEEDSamplerManual` but the cond-patching
    seam is explicit: the node calls LatentClass.prime before the first stage.
    """

    DESCRIPTION = (
        "Manual SPEED sampler (LatentClass edition) — set the stages by hand. "
        "Give up to four (goal, resolution) pairs: goal = step where that stage "
        "ends, resolution = scale (0.25 = quarter). Set goal or resolution to 0 "
        "to skip that stage."
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
                "noise_policy": (["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
                "ratio_mode": (list(RATIO_MODES), {"default": "steps"}),
                "transition_goal_1": ("FLOAT", {"default": 3, "min": 0, "max": 1000}),
                "transition_resolution_1": ("FLOAT", {"default": 0.25, "min": 0, "max": 1}),
                "transition_goal_2": ("FLOAT", {"default": 5, "min": 0, "max": 1000}),
                "transition_resolution_2": ("FLOAT", {"default": 0.5, "min": 0, "max": 1}),
                "transition_goal_3": ("FLOAT", {"default": 8, "min": 0, "max": 1000}),
                "transition_resolution_3": ("FLOAT", {"default": 0.75, "min": 0, "max": 1}),
                "transition_goal_4": ("FLOAT", {"default": 15, "min": 0, "max": 1000}),
                "transition_resolution_4": ("FLOAT", {"default": 1.0, "min": 0, "max": 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image,
               noise_policy="direct_coarse",
               seed_offset=10000,
               ratio_mode="steps",
               transition_goal_1=3, transition_resolution_1=0.25,
               transition_goal_2=5, transition_resolution_2=0.5,
               transition_goal_3=8, transition_resolution_3=0.75,
               transition_goal_4=15, transition_resolution_4=1.0, **kwargs):
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
        goals = [g for g, r in transitions if g > 0 and r != 0]

        if ratio_mode == "steps":
            step_goals = [int(g) for g in goals[:-1]]
        else:  # "ratio"
            step_goals = [int(round(g * total_steps)) for g in goals[:-1]]
        transition_steps = tuple(step_goals)

        validate_transition_steps(transition_steps, n_stages, len(sigmas))

        # Build the SpeedConfig from the live full-res dims.
        full_video, _ = unpack_latent(latent_image.get("samples"))
        config = SpeedConfig(
            scales=tuple(scales),
            transition_steps=tuple(transition_steps),
            transition_mode="explicit",
            noise_policy=noise_policy,
            delta=0.01,  # unused in explicit mode
            noise_amplitude=7.394,
            noise_decay_exponent=0.62,
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )

        # Prime the LatentClass registry before the first stage.
        LatentClass.prime(guider)

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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerManualClass": MiniMaxH3SPEEDSamplerManualClass}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerManualClass": "MiniMax H3 SPEED — Sampler (Manual, LatentClass)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSamplerManualClass"]
