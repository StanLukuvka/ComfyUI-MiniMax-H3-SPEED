"""Manual step-through SPEED sampler — explicit transition schedule.

Exposes the transition schedule directly: up to four
(transition_goal, transition_resolution) widget pairs.

Semantics:

- ``transition_goal_N == 0`` or ``transition_resolution_N == 0`` disables stage N.
- ``ratio_mode == "steps"`` (default): goal is a STEP INDEX (position in the
  sigma schedule) at which stage N ends; resolution is that stage's scale.
  The final active stage's goal is unused — it runs to the end of the
  schedule. Example (defaults): goals 3/5/8, resolutions 0.25/0.5/0.75/1.0
  == quarter → half → three-quarter → full.
- ``ratio_mode == "ratio"``: goal must be <= 1. Scale is
  ``resolution * goal``; the boundary is placed at ``round(goal * total_steps)``.
"""

from __future__ import annotations

from speed_scripts.config import RATIO_MODES
from speed_scripts.nodes_common import build_config_and_run, validate_transition_steps


def CALCULATE_SCALES(transitions, ratio_mode):
    """Build the per-stage scale factors from (goal, resolution) pairs.

    Steps mode: resolution is the stage's scale. Ratio mode: the goal is a
    fraction of the schedule, and the scale must be scaled by it. A goal of 0
    skips that stage; later stages stay active and shift down (the caller
    validates the resulting schedule).
    """
    scales = []
    for goal, resolution in transitions:
        if goal == 0 or resolution == 0:
            continue  # Skips this stage; later stages remain active.
        if ratio_mode == "steps":
            scales.append(resolution)
        elif ratio_mode == "ratio":
            if goal > 1:
                raise ValueError(f"Invalid goal for ratio mode: {goal}. Goal must be <= 1.")
            scales.append(resolution * goal)
    if not scales:
        raise ValueError("No valid scales calculated. Check transition goals and resolutions.")
    return scales


class MiniMaxH3SPEEDSamplerManual:
    """SPEED progressive-resolution diffusion for MiniMax-H3's packed latent.

    Manual step-through variant of MiniMaxH3SPEEDSampler: the transition
    schedule (per-stage boundaries and scales) is set directly on the node
    instead of coming from a named preset.
    """

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
        # Manual is explicit only — delta/A/beta are not used (steps are direct).
        # Keep reading kwargs for backwards compat (old workflows had Tolerance/A/beta widgets) but ignore them.
        _ = kwargs.get("Tolerance (Delta)", kwargs.get("delta", None))
        delta = 0.01  # unused in explicit, but SpeedConfig requires valid value
        noise_amplitude = 7.394
        noise_decay_exponent = 0.62

        transitions = [
            (float(transition_goal_1), float(transition_resolution_1)),
            (float(transition_goal_2), float(transition_resolution_2)),
            (float(transition_goal_3), float(transition_resolution_3)),
            (float(transition_goal_4), float(transition_resolution_4)),
        ]

        # Active stages: pairs with a nonzero goal. Interpretation depends on
        # ratio_mode (see CALCULATE_SCALES).
        scales = CALCULATE_SCALES(transitions, ratio_mode)
        n_stages = len(scales)
        if n_stages < 2:
            raise ValueError(
                "manual schedule needs at least two active stages (goal != 0) "
                "with the final stage at resolution 1.0"
            )

        total_steps = len(sigmas) - 1
        # Every active pair produced one scale, so goals and stages align 1:1.
        goals = [g for g, r in transitions if g > 0 and r != 0]

        # Boundary step for each active stage except the last (the last stage
        # runs to the end of the schedule, so its goal carries no boundary).
        if ratio_mode == "steps":
            step_goals = [int(g) for g in goals[:-1]]
        else:  # "ratio"
            step_goals = [int(round(g * total_steps)) for g in goals[:-1]]
        transition_steps = tuple(step_goals)

        # Steps-mode semantics: goal is the sigma-schedule step index of the
        # boundary. sanity: boundaries must be strictly increasing interior
        # indices (the same contract the preset DEFAULT_TRANSITION_STEPS hold).
        validate_transition_steps(transition_steps, n_stages, len(sigmas))

        return build_config_and_run(
            noise, guider, sigmas, latent_image,
            scales=scales,
            transition_steps=transition_steps,
            transition_mode="explicit",
            noise_policy=noise_policy,
            delta=delta,
            noise_amplitude=noise_amplitude,
            noise_decay_exponent=noise_decay_exponent,
            seed_offset=seed_offset,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerManual": MiniMaxH3SPEEDSamplerManual}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerManual": "MiniMax H3 SPEED — Sampler (Manual Step-Through)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSamplerManual"]
