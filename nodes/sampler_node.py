"""Automatic MiniMax-H3 SPEED sampler."""

from __future__ import annotations

import comfy.utils

from speed_scripts.families import MODEL_TYPES
from speed_scripts.runtime import run_speed_pipeline
from speed_scripts.planning import build_automatic_speed_config


class MiniMaxH3SPEEDSampler:
    """Progressive-resolution SPEED sampling with automatic stage placement."""

    DESCRIPTION = (
        "Automatic SPEED sampler — pick stages (2, 3, or 4) and go. "
        "Starts cheap at low resolution, then upsamples when the detail "
        "matters. Set Tolerance (0.5% = 0.005) to trade blur for speed. "
        "Uses baked Euler-derived A/beta; re-calibrate with the Harvest node "
        "if you change checkpoint, LoRA/addons, sampler, or another factor "
        "that materially changes the denoising trajectory."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    OUTPUT_TOOLTIPS = (
        "Final latent at full resolution, as from SamplerCustomAdvanced.",
        "Latent of the model's last clean-image (x0) estimate, as from SamplerCustomAdvanced.",
    )
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed"

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
                        "tooltip": "direct_coarse: start on coarse-size noise and fill each new frequency band from seeded noise at every transition. coupled_full_grid: take all bands from one full-resolution noise field drawn like SamplerCustomAdvanced, so it matches a plain run at the same seed, but on H3 video it gave back most of the speed gain (5-10% vs 15-30%).",
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
                        "default": 12.105,
                        "min": 0.0001,
                        "max": 1e6,
                        "step": 0.0001,
                        "round": 0.0001,
                        "tooltip": "A in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule; the default is Euler-derived for H3.",
                    },
                ),
                "noise_decay_exponent": (
                    "FLOAT",
                    {
                        "default": 0.773,
                        "min": 0.0001,
                        "max": 10.0,
                        "step": 0.0001,
                        "round": 0.0001,
                        "tooltip": "beta in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule; the default is Euler-derived for H3.",
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
            },
            "optional": {
                "model_type": (
                    list(MODEL_TYPES),
                    {
                        "default": "h3_video",
                        "tooltip": "h3_video: MiniMax-H3 nested video+audio latents. image: flow (CONST) image models such as MOSS Image or Krea 2. Must match the model; it is never auto-detected.",
                    },
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
        stages=3,
        noise_policy="direct_coarse",
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
        model_type="h3_video",
        **kwargs,
    ):
        delta = kwargs.pop("Tolerance (Delta)", 0.005)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Automatic option(s): {unexpected}")
        if isinstance(stages, bool) or not isinstance(stages, int) or stages not in (2, 3, 4):
            raise ValueError(f"stages must be exactly 2, 3, or 4; got {stages!r}")

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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3SPEEDSampler",
]
