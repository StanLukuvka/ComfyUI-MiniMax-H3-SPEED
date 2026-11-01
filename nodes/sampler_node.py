"""Automatic SPEED sampler for the h3_video and flow_image model types."""

from __future__ import annotations

import comfy.utils

from speed_scripts.node_widgets import (
    ADVANCED_BASE_INPUTS,
    MODEL_TYPE_WIDGET,
    SAMPLER_INPUT_AUTOMATIC,
    SPEED_AUTOMATIC_WIDGETS,
)
from speed_scripts.runtime import run_speed_pipeline
from speed_scripts.planning import build_automatic_speed_config


class SPEEDSampler:
    """Progressive-resolution SPEED sampling with automatic stage placement."""

    DESCRIPTION = (
        "Automatic SPEED sampler — pick stages (2, 3, or 4) and go. "
        "Starts cheap at low resolution, then upsamples when the detail "
        "matters. Set Tolerance (0.5% = 0.005) to trade blur for speed. "
        "Needs noise_amplitude and noise_decay_exponent from the Sigma Harvest "
        "node for your model; re-run Harvest if you change checkpoint, LoRA/addons, "
        "sampler, or another factor that materially changes the denoising trajectory."
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
        # Widget definitions live in speed_scripts/node_widgets.py; the key order is the
        # public contract (saved workflows store widget values by position).
        required = {
            "noise": ADVANCED_BASE_INPUTS["noise"],
            "guider": ADVANCED_BASE_INPUTS["guider"],
            "sampler": SAMPLER_INPUT_AUTOMATIC,
            "sigmas": ADVANCED_BASE_INPUTS["sigmas"],
            "latent_image": ADVANCED_BASE_INPUTS["latent_image"],
        }
        required.update(SPEED_AUTOMATIC_WIDGETS)
        required["model_type"] = MODEL_TYPE_WIDGET
        return {"required": required}

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        model_type,
        noise_amplitude,
        noise_decay_exponent,
        stages=3,
        noise_policy="direct_coarse",
        seed_offset=10000,
        **kwargs,
    ):
        delta = kwargs.pop("Tolerance (Delta)", 0.005)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Automatic option(s): {unexpected}")
        if isinstance(stages, bool) or not isinstance(stages, int) or stages not in (2, 3, 4):
            raise ValueError(f"stages must be exactly 2, 3, or 4; got {stages!r}")
        if not noise_amplitude or not noise_decay_exponent:
            raise ValueError(
                "noise_amplitude and noise_decay_exponent are not set (0). Run Sigma Harvest "
                "for this model, sampler and schedule and paste its values."
            )

        config = build_automatic_speed_config(
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
            sampler=sampler,
            model_type=model_type,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        )


class MiniMaxH3SPEEDSampler(SPEEDSampler):
    """Deprecated ID for ``SPEEDSampler``, kept so saved workflows still load."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "SPEEDSampler": SPEEDSampler,
    "MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDSampler": "SPEED — Sampler Advanced",
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler (deprecated)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSampler",
    "MiniMaxH3SPEEDSampler",
]
