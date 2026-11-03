"""Calibrate Automatic SPEED from one native full-resolution sampler run.

Measure `x - denoised` at each step, fit `P = A * |omega|^(-beta)`, and
return A, beta, and delta for the Automatic node. This measures the model's
residual spectrum, not the clean-data spectrum from the SPEED paper.
"""

from speed_scripts.harvest import run_harvest
from speed_scripts.node_widgets import SPEED_HARVEST_WIDGETS


class SPEEDSigmaHarvest:
    """Measure one native sampler run and return an Automatic calibration."""

    DESCRIPTION = (
        "Sigma Harvest calibrates Automatic from one native full-resolution "
        "sampler run. Re-run it when the sampler, model, scheduler, step count, "
        "or denoising behavior changes. It fits the residual spectrum "
        "(x - denoised) to P = A·|ω|^-beta and returns A/beta for Automatic. "
        "The calibration basis is the model's residual spectrum, separate from "
        "the clean-data spectrum in the SPEED paper."
    )
    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("calibration", "diagnostic_latent")
    OUTPUT_TOOLTIPS = (
        "Calibration JSON (noise_amplitude, noise_decay_exponent, delta, r2, health, report). Connect it to Preview Any to read it, then paste the values into the matching Automatic node. On failure, an error JSON.",
        "Latent from the native full-resolution pass, for inspection only. Not a SPEED result.",
    )
    FUNCTION = "harvest"
    # The node decides the model family: the Advanced nodes run MiniMax-H3.
    MODEL_TYPE = "h3_video"
    CATEGORY = "sampling/speed/diagnostics"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": (
                    "NOISE",
                    {"tooltip": "Same as SamplerCustomAdvanced; its seed also seeds the sampler."},
                ),
                "guider": (
                    "GUIDER",
                    {},
                ),
                "sampler": (
                    "SAMPLER",
                    {"tooltip": "Any ComfyUI SAMPLER (e.g. from KSamplerSelect). Use the sampler you will run SPEED with; the calibration is specific to it."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Sigma schedule for the native calibration pass. Use the same scheduler and step count you will run SPEED with."},
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "Latent for the native full-resolution pass, at the size you will run SPEED at."},
                ),
                **SPEED_HARVEST_WIDGETS,
            },
        }

    def harvest(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        **kwargs,
    ):

        delta = float(kwargs.pop("Tolerance (Delta)", 0.005))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Harvest option(s): {unexpected}")
        return run_harvest(noise, guider, sampler, sigmas, latent_image, self.MODEL_TYPE, delta)

class MiniMaxH3HarvestToConfig(SPEEDSigmaHarvest):
    """Deprecated ID for ``SPEEDSigmaHarvest``, kept so saved workflows still load."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "SPEEDSigmaHarvest": SPEEDSigmaHarvest,
    "MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDSigmaHarvest": "SPEED — Sigma Harvest Advanced",
    "MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Sigma Harvest (Native Sampler) (deprecated)",
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSigmaHarvest",
    "MiniMaxH3HarvestToConfig",
]
