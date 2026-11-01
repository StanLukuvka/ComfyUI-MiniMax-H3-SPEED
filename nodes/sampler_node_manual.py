"""Manual SPEED sampler with an explicit stage schedule."""

from __future__ import annotations

import comfy.utils

from speed_scripts.node_widgets import (
    ADVANCED_BASE_INPUTS,
    SAMPLER_INPUT_MANUAL,
    SPEED_MANUAL_WIDGETS,
)
from speed_scripts.runtime import run_speed_pipeline
from speed_scripts.planning import build_manual_speed_config


class SPEEDSamplerManual:
    """Progressive-resolution SPEED sampling with user-defined stage boundaries."""

    DESCRIPTION = (
        "Manual SPEED sampler — set up to four (goal, resolution) pairs by hand. "
        "For every stage except the last active one, goal marks where that stage "
        "ends and resolution is its scale (0.25 = quarter). The final active stage "
        "always runs to the end of the sigma schedule. Set goal or resolution to 0 "
        "to skip a stage. Use this to copy known schedules or test custom ladders."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    OUTPUT_TOOLTIPS = (
        "Final latent at full resolution.",
        "Latent of the model's last clean-image (x0) estimate, as from SamplerCustomAdvanced.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/speed"
    # The node decides the model family: the Advanced nodes run MiniMax-H3.
    MODEL_TYPE = "h3_video"

    @classmethod
    def INPUT_TYPES(cls):
        # Widget definitions live in speed_scripts/node_widgets.py; the key order is the
        # public contract (saved workflows store widget values by position).
        required = {
            "noise": ADVANCED_BASE_INPUTS["noise"],
            "guider": ADVANCED_BASE_INPUTS["guider"],
            "sampler": SAMPLER_INPUT_MANUAL,
            "sigmas": ADVANCED_BASE_INPUTS["sigmas"],
            "latent_image": ADVANCED_BASE_INPUTS["latent_image"],
        }
        required.update(SPEED_MANUAL_WIDGETS)
        return {"required": required}

    def sample(
        self,
        noise,
        guider,
        sampler,
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
            sampler=sampler,
            model_type=self.MODEL_TYPE,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )


class MiniMaxH3SPEEDSamplerManual(SPEEDSamplerManual):
    """Deprecated ID for ``SPEEDSamplerManual``, kept so saved workflows still load."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "SPEEDSamplerManual": SPEEDSamplerManual,
    "MiniMaxH3SPEEDSamplerManual": MiniMaxH3SPEEDSamplerManual,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDSamplerManual": "SPEED — Sampler Advanced (Manual)",
    "MiniMaxH3SPEEDSamplerManual": "MiniMax H3 SPEED — Sampler (Manual Step-Through) (deprecated)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSamplerManual",
    "MiniMaxH3SPEEDSamplerManual",
]
