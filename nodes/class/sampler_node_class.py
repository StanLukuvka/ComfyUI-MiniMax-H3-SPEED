"""Automatic SPEED sampler — uses LatentClass directly (no build_config_and_run).

Picks 2-4 resolution stages (0.5→1.0, 0.33→0.66→1.0, 0.25→0.5→0.75→1.0).
Steps are placed automatically from Tolerance + A/beta via the power-spectrum
threshold. The cond-patching is done via LatentClass.walk_guider — the latent
lifecycle is owned by LatentClass, not embedded in h3_runtime.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import (
    run_speed_pipeline,
    unpack_latent,
)
from speed_scripts.latent_class import LatentClass


# Stages -> scale ladder for Automatic. Evenly spaced, ends at 1.0.
# 2: 0.5 → 1.0, 3: 0.33 → 0.66 → 1.0, 4: 0.25 → 0.5 → 0.75 → 1.0
STAGES_TO_SCALES: dict[int, tuple[float, ...]] = {
    2: (0.5, 1.0),
    3: (0.3333333333, 0.6666666667, 1.0),
    4: (0.25, 0.5, 0.75, 1.0),
}
# Backwards compat: old preset names -> stages (for workflows saved before the rename)
PRESET_TO_STAGES: dict[str, int] = {
    "half_then_full": 2,
    "three_quarter_then_full": 2,
    "quarter_half_full": 3,
    "aggressive": 3,
    "quarter_half_3q_full": 4,
}


class MiniMaxH3SPEEDSamplerClass:
    """SPEED progressive-resolution diffusion for MiniMax-H3's packed latent.

    Same as the original `MiniMaxH3SPEEDSampler` but the cond-patching
    seam is explicit: the node calls `LatentClass.walk_guider` to drive
    every keyframe + ref2va ref through the SPEED stage boundaries
    instead of relying on h3_runtime's internal call.
    """

    DESCRIPTION = (
        "Automatic SPEED sampler (LatentClass edition) — pick stages (2, 3, or 4) "
        "and go. Starts cheap at low resolution, then upsamples when the detail "
        "matters. Set Tolerance (1% = 0.01) to trade blur for speed. Uses baked "
        "A/beta; re-calibrate with the Harvest node if you change checkpoint."
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
                "Tolerance (Delta)": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "noise_amplitude": ("FLOAT", {"default": 7.394, "min": 0.0, "max": 1e6}),
                "noise_decay_exponent": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, stages=3,
               noise_policy="direct_coarse",
               noise_amplitude=7.394, noise_decay_exponent=0.62,
               seed_offset=10000, **kwargs):
        # Tolerance (Delta) is the UI label — accept delta alias for old workflows/tests
        delta = kwargs.get("Tolerance (Delta)",
                kwargs.get("Tolerance",
                kwargs.get("tolerance",
                kwargs.get("delta", kwargs.get("Delta", 0.01)))))
        delta = float(delta)
        if "preset" in kwargs:
            preset = kwargs.pop("preset")
            stages = PRESET_TO_STAGES.get(preset, stages)
        try:
            stages = int(stages)
        except Exception:
            stages = 3
        stages = max(2, min(4, stages))
        scales = STAGES_TO_SCALES[stages]
        # Dummy steps — validated then overridden by delta_custom power-spectrum thresholds
        transition_steps = tuple(range(1, len(scales)))

        # Resolve the live full-res dims, build the SpeedConfig.
        full_video, _ = unpack_latent(latent_image.get("samples"))
        config = SpeedConfig(
            scales=tuple(scales),
            transition_steps=tuple(transition_steps),
            transition_mode="delta_custom",
            noise_policy=noise_policy,
            delta=float(delta),
            noise_amplitude=float(noise_amplitude),
            noise_decay_exponent=float(noise_decay_exponent),
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )

        # Snapshot pristine for every keyframe/ref on the guider before the
        # first stage boundary. The runtime will call walk_guider again at
        # every boundary (via the h3_runtime shim) to do the actual resize.
        LatentClass.prime(guider)

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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerClass": MiniMaxH3SPEEDSamplerClass}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerClass": "MiniMax H3 SPEED — Sampler (LatentClass)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSamplerClass"]
