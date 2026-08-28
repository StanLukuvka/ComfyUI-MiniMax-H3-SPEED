"""Automatic SPEED sampler test variant — class renamed with `_test` suffix.

Same body as `sampler_node_class.py`; the `_test` suffix lets the test suite
import it without colliding with the live node name. The class still calls
`LatentClass.prime` and `LatentClass.walk_guider` directly — the test target
is the LatentClass path, not build_config_and_run.
"""

from __future__ import annotations

import comfy.samplers

from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import (
    run_speed_pipeline,
    unpack_latent,
)
from speed_scripts.latent_class import LatentClass


STAGES_TO_SCALES: dict[int, tuple[float, ...]] = {
    2: (0.5, 1.0),
    3: (0.3333333333, 0.6666666667, 1.0),
    4: (0.25, 0.5, 0.75, 1.0),
}
PRESET_TO_STAGES: dict[str, int] = {
    "half_then_full": 2,
    "three_quarter_then_full": 2,
    "quarter_half_full": 3,
    "aggressive": 3,
    "quarter_half_3q_full": 4,
}


class MiniMaxH3SPEEDSamplerClassTest:
    """Test variant of the LatentClass-edition automatic sampler.

    Behaviour identical to `MiniMaxH3SPEEDSamplerClass`. The `_test` suffix
    keeps it out of the production NODE_CLASS_MAPPINGS export so workflow
    files don't accidentally pick it up.
    """

    DESCRIPTION = (
        "TEST variant: Automatic SPEED sampler (LatentClass edition). "
        "Same body as the live class edition — exposed for tests only."
    )
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "sampling/minimax_h3_speed/_test"

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
        transition_steps = tuple(range(1, len(scales)))

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


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSamplerClassTest": MiniMaxH3SPEEDSamplerClassTest}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SPEEDSamplerClassTest": "MiniMax H3 SPEED — Sampler (LatentClass) [TEST]"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSamplerClassTest"]
