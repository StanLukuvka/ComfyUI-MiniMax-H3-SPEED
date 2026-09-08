"""SPEED Sigma Harvest (Continuous) — diagnostic node.

Runs ONE real multi-stage SPEED generation (the same `run_speed_pipeline` the
Automatic sampler uses, with the same shared config builder) and collects
per-callback spectral telemetry through the runtime observer. Observational
only: the generation is bit-for-bit identical to a run without the collector
(the observer from commit 2 is the only runtime surface, and attaching one is
behaviorally inert — proven by the commit-2 test suite).

The collector lives in `speed_scripts.online_harvest`; this file is node
wiring only.
"""

from __future__ import annotations

import comfy.samplers
import comfy.utils

from speed_scripts.automatic_config import (
    PRESET_TO_STAGES,
    build_automatic_speed_config,
)
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.latent_class import LatentWalker
from speed_scripts.online_harvest import SpeedHarvestCollector


class MiniMaxH3SPEEDSigmaHarvest:
    """Continuous SPEED Sigma Harvest — one SPEED run + per-step spectra.

    Takes the Automatic sampler's generation inputs (same defaults, same
    shared `build_automatic_speed_config`) plus a small diagnostic block,
    and returns the harvest JSON alongside the normal SPEED outputs.
    """

    DESCRIPTION = (
        "SPEED Sigma Harvest (Continuous) — runs a normal SPEED generation "
        "but records the model's spectral state at every denoising step. "
        "Reports both the residual spectrum used by this project's empirical "
        "calibration and the denoised/x0 spectrum used by SPEED's theoretical "
        "power-spectrum model. Observational only: stage transitions are not "
        "moved during generation."
    )
    RETURN_TYPES = ("STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("harvest_json", "output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed/diagnostics"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "stages": ("INT", {"default": 3, "min": 2, "max": 4}),
                "noise_policy": (["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"}),
                "Tolerance (Delta)": ("FLOAT", {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "noise_amplitude": ("FLOAT", {"default": 12.105, "min": 0.0, "max": 1e6, "step": 0.001, "round": 0.001}),
                "noise_decay_exponent": ("FLOAT", {"default": 0.773, "min": 0.0, "max": 10.0, "step": 0.001, "round": 0.001}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
                "measurement_mode": (["both", "x0_only", "residual_only"], {"default": "both"}),
                "analysis_stride": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "smoothing_alpha": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01}),
                "boundary_band_half_width": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 32.0, "step": 0.01}),
                "store_radial_profiles": ("BOOLEAN", {"default": False}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, stages=3,
               noise_policy="direct_coarse",
               noise_amplitude=12.105, noise_decay_exponent=0.773,
               seed_offset=10000,
               measurement_mode="both", analysis_stride=1,
               smoothing_alpha=0.25, boundary_band_half_width=1.0,
               store_radial_profiles=False, **kwargs):
        # Tolerance (Delta) is the UI label — same alias chain as the
        # Automatic sampler so old workflows behave identically.
        delta = kwargs.get("Tolerance (Delta)",
                kwargs.get("Tolerance",
                kwargs.get("tolerance",
                kwargs.get("delta", kwargs.get("Delta", 0.005)))))
        if "preset" in kwargs:
            preset = kwargs.pop("preset")
            stages = PRESET_TO_STAGES.get(preset, stages)
        try:
            stages = int(stages)
        except Exception:
            stages = 3
        stages = max(2, min(4, stages))
        config = build_automatic_speed_config(
            latent_image,
            stages=stages,
            noise_policy=noise_policy,
            delta=delta,
            noise_amplitude=noise_amplitude,
            noise_decay_exponent=noise_decay_exponent,
            seed_offset=seed_offset,
        )

        collector = SpeedHarvestCollector(
            delta=config.delta,
            noise_amplitude=config.noise_amplitude,
            noise_decay_exponent=config.noise_decay_exponent,
            measurement_mode=measurement_mode,
            analysis_stride=analysis_stride,
            smoothing_alpha=smoothing_alpha,
            boundary_band_half_width=boundary_band_half_width,
            store_radial_profiles=store_radial_profiles,
        )
        collector.set_original_sigmas(sigmas)

        # Same walker priming as the Automatic sampler: snapshot pristine
        # keyframe/ref latents before the first stage boundary.
        LatentWalker(guider)

        output, denoised = run_speed_pipeline(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            sampler=comfy.samplers.sampler_object("euler"),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
            observer=collector,
        )
        return (collector.to_json(), output, denoised)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSigmaHarvest": MiniMaxH3SPEEDSigmaHarvest}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSigmaHarvest": "MiniMax H3 SPEED — SPEED Sigma Harvest (Continuous)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSigmaHarvest"]
