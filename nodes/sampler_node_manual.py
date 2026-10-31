"""Manual SPEED sampler with an explicit stage schedule."""

from __future__ import annotations

import comfy.utils

from speed_scripts.config import RATIO_MODES
from speed_scripts.families import MODEL_TYPES
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
        "Final latent at full resolution, as from SamplerCustomAdvanced.",
        "Latent of the model's last clean-image (x0) estimate, as from SamplerCustomAdvanced.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/speed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": (
                    "NOISE",
                    {"tooltip": "Noise source, same as SamplerCustomAdvanced (e.g. RandomNoise). Its seed also seeds the sampler."},
                ),
                "guider": (
                    "GUIDER",
                    {"tooltip": "Guider, same as SamplerCustomAdvanced (e.g. CFGGuider, BasicGuider)."},
                ),
                "sampler": (
                    "SAMPLER",
                    {"tooltip": "Any ComfyUI SAMPLER (e.g. from KSamplerSelect). Any sampler works; Euler is the one SPEED was measured with."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Sigma schedule, same as SamplerCustomAdvanced. Stage boundaries are placed on this schedule."},
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "Latent to sample, same as SamplerCustomAdvanced. The final stage runs at this full size."},
                ),
                "noise_policy": (
                    ["direct_coarse", "coupled_full_grid"],
                    {
                        "default": "direct_coarse",
                        "tooltip": "direct_coarse: start on coarse-size noise and fill each new frequency band from seeded noise at every transition. coupled_full_grid: take all bands from one full-resolution noise field drawn like SamplerCustomAdvanced, so it matches a plain run at the same seed. On H3 video it saved only 5-10%, against 15-30% for direct_coarse.",
                    },
                ),
                "seed_offset": (
                    "INT",
                    {
                        "default": 10000,
                        "min": 0,
                        "max": 2**31 - 1,
                        "tooltip": "direct_coarse only: salts the seed of the high-frequency fill drawn at each resolution transition (seed + seed_offset + stage index). Ignored by coupled_full_grid. Does not change the first stage's noise or the sampler seed. Leave at 10000 unless you want a different fill pattern for the same seed.",
                    },
                ),
                "ratio_mode": (
                    list(RATIO_MODES),
                    {
                        "default": "steps",
                        "tooltip": "How transition goals are read. steps: whole step indices into the sigma schedule. ratio: 0-1 fractions of the schedule, placed at round(goal * total steps).",
                    },
                ),
                "transition_goal_1": (
                    "FLOAT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Where stage 1 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
                    },
                ),
                "transition_resolution_1": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": 0,
                        "max": 1,
                        "tooltip": "Stage 1 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
                    },
                ),
                "transition_goal_2": (
                    "FLOAT",
                    {
                        "default": 5,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Where stage 2 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
                    },
                ),
                "transition_resolution_2": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0,
                        "max": 1,
                        "tooltip": "Stage 2 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
                    },
                ),
                "transition_goal_3": (
                    "FLOAT",
                    {
                        "default": 8,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Where stage 3 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
                    },
                ),
                "transition_resolution_3": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0,
                        "max": 1,
                        "tooltip": "Stage 3 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
                    },
                ),
                "transition_goal_4": (
                    "FLOAT",
                    {
                        "default": 15,
                        "min": 0,
                        "max": 1000,
                        "tooltip": "Where stage 4 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
                    },
                ),
                "transition_resolution_4": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0,
                        "max": 1,
                        "tooltip": "Stage 4 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
                    },
                ),
                "model_type": (
                    list(MODEL_TYPES),
                    {"tooltip": "Which model family this is. h3_video: MiniMax-H3 nested video+audio latents. flow_image: flow (CONST) image models such as MOSS Image or Krea 2. Required, never auto-detected; a value that does not match the model and latent stops with an error."},
                ),
            },
        }

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        model_type,
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
            model_type=model_type,
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
    "SPEEDSamplerManual": "SPEED — Sampler (Manual Step-Through)",
    "MiniMaxH3SPEEDSamplerManual": "MiniMax H3 SPEED — Sampler (Manual Step-Through) (deprecated)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSamplerManual",
    "MiniMaxH3SPEEDSamplerManual",
]
