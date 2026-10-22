"""Automatic SPEED sampler for the h3_video and image model types."""

from __future__ import annotations

import comfy.utils

from speed_scripts.families import MODEL_TYPES
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
                    {"tooltip": "Any ComfyUI SAMPLER (e.g. from KSamplerSelect). Euler is the calibrated reference; other samplers work but their baked numbers are unmeasured."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Sigma schedule, same as SamplerCustomAdvanced. Stage boundaries are placed on this schedule."},
                ),
                "latent_image": (
                    "LATENT",
                    {"tooltip": "Latent to sample, same as SamplerCustomAdvanced. The final stage runs at this full size."},
                ),
                "stages": (
                    "INT",
                    {
                        "default": 3,
                        "min": 2,
                        "max": 4,
                        "tooltip": "Number of progressive-resolution stages (2-4). 2 = 0.5x then full; 3 = 1/3x, 2/3x, full; 4 = 0.25x, 0.5x, 0.75x, full.",
                    },
                ),
                "noise_policy": (
                    ["direct_coarse", "coupled_full_grid"],
                    {
                        "default": "direct_coarse",
                        "tooltip": "direct_coarse: start on coarse-size noise and fill each new frequency band from seeded noise at every transition. coupled_full_grid: take all bands from one full-resolution noise field drawn like SamplerCustomAdvanced, so it matches a plain run at the same seed. On H3 video it saved only 5-10%, against 15-30% for direct_coarse.",
                    },
                ),
                "Tolerance (Delta)": (
                    "FLOAT",
                    {
                        "default": 0.005,
                        "min": 1e-4,
                        "max": 0.5,
                        "step": 0.001,
                        "tooltip": "Delta: how much error is accepted when a stage switches to a higher resolution. Smaller = more conservative, later transitions, smaller saving. Boundaries come from the power-spectrum fit (noise_amplitude, noise_decay_exponent). For images, 0.05 is the useful setting.",
                    },
                ),
                "noise_amplitude": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1e6,
                        "step": 0.0001,
                        "round": 0.0001,
                        "tooltip": "A in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule. There is no default: 0 means not set and stops with an error.",
                    },
                ),
                "noise_decay_exponent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "round": 0.0001,
                        "tooltip": "beta in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule. There is no default: 0 means not set and stops with an error.",
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
                "model_type": (
                    list(MODEL_TYPES),
                    {"tooltip": "Which model family this is. h3_video: MiniMax-H3 nested video+audio latents. image: flow (CONST) image models such as MOSS Image or Krea 2. Required, never auto-detected; a value that does not match the model and latent stops with an error."},
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
    "SPEEDSampler": "SPEED — Sampler",
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler (deprecated)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "SPEEDSampler",
    "MiniMaxH3SPEEDSampler",
]
