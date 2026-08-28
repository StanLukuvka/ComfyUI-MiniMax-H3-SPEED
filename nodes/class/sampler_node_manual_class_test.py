"""Manual step-through SPEED sampler test variant — class renamed with `_test` suffix.

Same body as `sampler_node_manual_class.py`; the `_test` suffix keeps the test
import out of the live NODE_CLASS_MAPPINGS.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.config import RATIO_MODES, SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline, unpack_latent
from speed_scripts.latent_class import LatentWalker
from speed_scripts.nodes_common import validate_transition_steps


def CALCULATE_SCALES(transitions, ratio_mode):
    """Build the per-stage scale factors from (goal, resolution) pairs."""
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


class MiniMaxH3SPEEDSamplerManualClassTest:
    """Test variant of the LatentClass-edition manual sampler.

    Behaviour identical to `MiniMaxH3SPEEDSamplerManualClass`. The `_test` suffix
    keeps it out of the production NODE_CLASS_MAPPINGS export.
    """

    DESCRIPTION = (
        "TEST variant: Manual SPEED sampler (LatentClass edition). "
        "Same body as the live class edition — exposed for tests only."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed/_test"

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
        else:
            step_goals = [int(round(g * total_steps)) for g in goals[:-1]]
        transition_steps = tuple(step_goals)

        validate_transition_steps(transition_steps, n_stages, len(sigmas))

        full_video, _ = unpack_latent(latent_image.get("samples"))
        config = SpeedConfig(
            scales=tuple(scales),
            transition_steps=tuple(transition_steps),
            transition_mode="explicit",
            noise_policy=noise_policy,
            delta=0.01,
            noise_amplitude=7.394,
            noise_decay_exponent=0.62,
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )

        LatentWalker(guider)

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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerManualClassTest": MiniMaxH3SPEEDSamplerManualClassTest}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerManualClassTest": "MiniMax H3 SPEED — Sampler (Manual, LatentClass) [TEST]"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSamplerManualClassTest"]
