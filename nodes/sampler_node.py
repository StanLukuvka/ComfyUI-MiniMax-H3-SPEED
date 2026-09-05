"""Automatic SPEED sampler — uses LatentWalker to own the I2V keyframe
lifecycle across the SPEED stage boundaries.

Picks 2-4 resolution stages (0.5→1.0, 0.33→0.66→1.0, 0.25→0.5→0.75→1.0).
Steps are placed automatically from Tolerance + A/beta via the power-spectrum
threshold. The cond-patching is done via LatentWalker — the latent lifecycle
is owned by the walker, not embedded in h3_runtime.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.latent_class import LatentWalker
from speed_scripts.nodes_common import _build_preview_callback, full_res_dims


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
        full_h, full_w = full_res_dims(latent_image)
        config = SpeedConfig(
            scales=tuple(scales),
            transition_steps=tuple(transition_steps),
            transition_mode="delta_custom",
            noise_policy=noise_policy,
            delta=float(delta),
            noise_amplitude=float(noise_amplitude),
            noise_decay_exponent=float(noise_decay_exponent),
            transition_seed_offset=int(seed_offset),
            full_latent_h=full_h,
            full_latent_w=full_w,
        )

        # Snapshot pristine for every keyframe/ref on the guider before the
        # first stage boundary. The runtime will call apply_stage again at
        # every boundary (via the h3_runtime shim) to do the actual resize.
        LatentWalker(guider)

        # Build ComfyUI's native preview callback at the node layer (matches
        # `SamplerCustomAdvanced.execute`): one x0_output dict shared across
        # the whole run, one callback built for the full step total. The
        # runtime wraps this per stage so the bar's `step+1` value advances
        # continuously instead of resetting each stage.
        x0_output: dict = {}
        preview_callback = _build_preview_callback(
            guider, len(sigmas) - 1, x0_output,
        )

        return run_speed_pipeline(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            sampler=comfy.samplers.sampler_object("euler"),
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
            preview_callback=preview_callback,
            x0_output=x0_output,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler"]
