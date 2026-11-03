"""Manual KSampler-style SPEED node: native KSampler widgets plus a stage schedule."""

from __future__ import annotations

from sampler_node_manual import SPEEDSamplerManual
from speed_scripts.ksampler_inputs import build_ksampler_inputs
from speed_scripts.node_widgets import (
    KSAMPLER_BASE_INPUTS,
    SPEED_MANUAL_WIDGETS,
)


class SPEEDKSamplerManual:
    """Progressive-resolution SPEED sampling with user-defined stage boundaries."""

    DESCRIPTION = SPEEDSamplerManual.DESCRIPTION
    RETURN_TYPES = SPEEDSamplerManual.RETURN_TYPES
    RETURN_NAMES = SPEEDSamplerManual.RETURN_NAMES
    OUTPUT_TOOLTIPS = SPEEDSamplerManual.OUTPUT_TOOLTIPS
    FUNCTION = "sample"
    CATEGORY = "sampling/speed"
    # The node decides the model family: the KSampler nodes run flow image models.
    MODEL_TYPE = "flow_image"

    @classmethod
    def INPUT_TYPES(cls):
        # KSampler base inputs, then the Manual SPEED widgets; key order is
        # the public contract.
        required = dict(KSAMPLER_BASE_INPUTS)
        required.update(SPEED_MANUAL_WIDGETS)
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
        **speed_options,
    ):
        # Build exactly what KSampler builds internally, then run the Advanced
        # Manual body on it: identical schedule validation, config and pipeline.
        noise, guider, sampler, sigmas = build_ksampler_inputs(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        )
        if len(sigmas) < 2:
            # denoise <= 0 leaves no sampling steps; ComfyUI's KSampler returns
            # the latent unchanged, so no schedule normalization or pipeline runs.
            return latent_image, latent_image
        # Unbound call so the Advanced body runs with this node's MODEL_TYPE.
        return SPEEDSamplerManual.sample(self, noise, guider, sampler, sigmas, latent_image, **speed_options)


NODE_CLASS_MAPPINGS = {
    "SPEEDKSamplerManual": SPEEDKSamplerManual,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDKSamplerManual": "SPEED — KSampler (Manual)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDKSamplerManual",
]
