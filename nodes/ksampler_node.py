"""Automatic KSampler-style SPEED node: native KSampler widgets plus SPEED options."""

from __future__ import annotations

from sampler_node import SPEEDSampler
from speed_scripts.ksampler_inputs import build_ksampler_inputs
from speed_scripts.node_widgets import (
    KSAMPLER_BASE_INPUTS,
    MODEL_TYPE_WIDGET,
    SPEED_AUTOMATIC_WIDGETS,
)


class SPEEDKSampler:
    """Progressive-resolution SPEED sampling with automatic stage placement."""

    DESCRIPTION = SPEEDSampler.DESCRIPTION
    RETURN_TYPES = SPEEDSampler.RETURN_TYPES
    RETURN_NAMES = SPEEDSampler.RETURN_NAMES
    OUTPUT_TOOLTIPS = SPEEDSampler.OUTPUT_TOOLTIPS
    FUNCTION = "sample"
    CATEGORY = "sampling/speed"

    @classmethod
    def INPUT_TYPES(cls):
        # KSampler base inputs, then model_type right after denoise, then the
        # Automatic SPEED widgets; key order is the public contract.
        required = dict(KSAMPLER_BASE_INPUTS)
        required["model_type"] = MODEL_TYPE_WIDGET
        required.update(SPEED_AUTOMATIC_WIDGETS)
        return {"required": required}

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        model_type,
        **speed_options,
    ):
        # Build exactly what KSampler builds internally, then run the Advanced
        # Automatic body on it: identical validation, config and pipeline call.
        noise, guider, sampler, sigmas = build_ksampler_inputs(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        )
        if len(sigmas) < 2:
            # denoise <= 0 leaves no sampling steps; ComfyUI's KSampler returns
            # the latent unchanged, so no SPEED validation or pipeline runs.
            return latent_image, latent_image
        return SPEEDSampler().sample(
            noise, guider, sampler, sigmas, latent_image, model_type=model_type, **speed_options
        )


NODE_CLASS_MAPPINGS = {
    "SPEEDKSampler": SPEEDKSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDKSampler": "SPEED — KSampler",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDKSampler",
]
