"""Calibrate Automatic SPEED from one native full-resolution sampler run.

Measure `x - denoised` at each step, fit `P = A * |omega|^(-beta)`, and
return A, beta, and delta for the Automatic node. This measures the model's
residual spectrum, not the clean-data spectrum from the SPEED paper.
"""

from speed_scripts.families import MODEL_TYPES
from speed_scripts.harvest import run_harvest


class SPEEDSigmaHarvest:
    """Measure one native sampler run and return an Automatic calibration."""

    DESCRIPTION = (
        "Sigma Harvest calibrates Automatic from one native full-resolution "
        "sampler run. Re-run it when the sampler, model, scheduler, step count, "
        "or denoising behavior changes. It fits the residual spectrum "
        "(x - denoised) to P = A·|ω|^-beta and returns A/beta for Automatic. "
        "The calibration basis is the model's residual spectrum, separate from "
        "the clean-data spectrum in the SPEED paper. Set model_type to match "
        "the SPEED Sampler."
    )
    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("calibration", "diagnostic_latent")
    OUTPUT_TOOLTIPS = (
        "Calibration JSON (noise_amplitude, noise_decay_exponent, delta, r2, health, report). Connect it to Preview Any to read it, then paste the values into the matching Automatic node. On failure, an error JSON.",
        "Latent from the native full-resolution pass, for inspection only. Not a SPEED result.",
    )
    FUNCTION = "harvest"
    CATEGORY = "sampling/speed/diagnostics"
    OUTPUT_NODE = False

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
                "Tolerance (Delta)": (
                    "FLOAT",
                    {
                        "default": 0.005,
                        "min": 1e-4,
                        "max": 0.5,
                        "step": 0.001,
                        "tooltip": "The Delta the report computes transition boundaries for. It is written into the calibration and the paste line; the fit itself does not depend on it.",
                    },
                ),
                "model_type": (
                    list(MODEL_TYPES),
                    {"tooltip": "Which model family this is. h3_video: MiniMax-H3 nested video+audio latents. flow_image: flow (CONST) image models such as MOSS Image or Krea 2. Required, never auto-detected; a value that does not match the model and latent stops with an error."},
                ),
            },
        }

    def harvest(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        model_type,
        **kwargs,
    ):

        delta = float(kwargs.pop("Tolerance (Delta)", 0.005))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Harvest option(s): {unexpected}")
        return run_harvest(noise, guider, sampler, sigmas, latent_image, model_type, delta)

class MiniMaxH3HarvestToConfig(SPEEDSigmaHarvest):
    """Deprecated ID for ``SPEEDSigmaHarvest``, kept so saved workflows still load."""

    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "SPEEDSigmaHarvest": SPEEDSigmaHarvest,
    "MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SPEEDSigmaHarvest": "SPEED — Sigma Harvest (Native Sampler)",
    "MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Sigma Harvest (Native Sampler) (deprecated)",
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSigmaHarvest",
    "MiniMaxH3HarvestToConfig",
]
