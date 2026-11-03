"""KSampler-style Sigma Harvest: native KSampler widgets for the calibration pass."""

from __future__ import annotations

from sampler_sigma_harvest_node import SPEEDSigmaHarvest
from speed_scripts.harvest import run_harvest
from speed_scripts.ksampler_inputs import build_ksampler_inputs
from speed_scripts.node_widgets import (
    KSAMPLER_BASE_INPUTS,
    SPEED_HARVEST_WIDGETS,
)


class SPEEDKSamplerHarvest:
    """Measure one native sampler run and return an Automatic calibration."""

    DESCRIPTION = SPEEDSigmaHarvest.DESCRIPTION
    RETURN_TYPES = SPEEDSigmaHarvest.RETURN_TYPES
    RETURN_NAMES = SPEEDSigmaHarvest.RETURN_NAMES
    OUTPUT_TOOLTIPS = SPEEDSigmaHarvest.OUTPUT_TOOLTIPS
    FUNCTION = "harvest"
    # The node decides the model family: the KSampler nodes run flow image models.
    MODEL_TYPE = "flow_image"
    CATEGORY = "sampling/speed/diagnostics"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # KSampler base inputs without denoise (Harvest always calibrates on
        # the full schedule), then the Harvest widgets; key order is the
        # public contract.
        required = dict(KSAMPLER_BASE_INPUTS)
        del required["denoise"]
        required.update(SPEED_HARVEST_WIDGETS)
        return {"required": required}

    def harvest(
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
        **kwargs,
    ):
        delta = float(kwargs.pop("Tolerance (Delta)", 0.005))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Harvest option(s): {unexpected}")
        # Build exactly what KSampler builds internally, then run the Advanced
        # Harvest body on it. Harvest always calibrates on the full schedule,
        # so there is no denoise widget: denoise=1.0.
        noise, guider, sampler, sigmas = build_ksampler_inputs(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise=1.0
        )
        return run_harvest(noise, guider, sampler, sigmas, latent_image, self.MODEL_TYPE, delta)


NODE_CLASS_MAPPINGS = {
    "SPEEDKSamplerHarvest": SPEEDKSamplerHarvest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDKSamplerHarvest": "SPEED — KSampler Sigma Harvest",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDKSamplerHarvest",
]
