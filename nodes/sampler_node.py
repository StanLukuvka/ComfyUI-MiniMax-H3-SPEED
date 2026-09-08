# FLOW-PRODUCED — Implementation Plan — Continuous SPEED Sigma Harvester.md §7 (commit 1) — flow-produced, do not hand-edit
"""Automatic SPEED sampler — uses LatentWalker to own the I2V keyframe
lifecycle across the SPEED stage boundaries.

Picks 2-4 resolution stages (0.5→1.0, 0.33→0.66→1.0, 0.25→0.5→0.75→1.0).
Steps are placed automatically from Tolerance + A/beta via the power-spectrum
threshold. The cond-patching is done via LatentWalker — the latent lifecycle
is owned by the walker, not embedded in h3_runtime.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.automatic_config import (
    PRESET_TO_STAGES,
    build_automatic_speed_config,
)
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.latent_class import LatentWalker


class MiniMaxH3SPEEDSampler:
    """SPEED progressive-resolution diffusion for MiniMax-H3's packed latent.

    Drop-in replacement for the standard KSAMPLER + SamplerCustomAdvanced
    pair. Takes (noise, guider, sigmas, latent_image) and runs a multi-stage
    diffusion that starts cheap at low resolution and upsamples when the
    detail matters. Steps per stage are placed automatically from the
    power-spectrum threshold (Tolerance + A/beta) so the user just picks
    "how many stages" and goes.
    """

    DESCRIPTION = (
        "Automatic SPEED sampler — pick stages (2, 3, or 4) and go. "
        "Starts cheap at low resolution, then upsamples when the detail "
        "matters. Set Tolerance (1% = 0.01) to trade blur for speed. "
        "Uses baked A/beta; re-calibrate with the Harvest node if you "
        "change checkpoint."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed"

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
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, stages=3,
               noise_policy="direct_coarse",
               noise_amplitude=12.105, noise_decay_exponent=0.773,
               seed_offset=10000, **kwargs):
        # Tolerance (Delta) is the UI label — accept delta alias for old workflows/tests
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

        # Snapshot pristine for every keyframe/ref on the guider before the
        # first stage boundary. The runtime will call apply_stage again at
        # every boundary (via the h3_runtime shim) to do the actual resize.
        LatentWalker(guider)

        return run_speed_pipeline(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            sampler=comfy.samplers.sampler_object("euler"),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler"]
