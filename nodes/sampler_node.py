"""Automatic SPEED sampler — simple English.

Picks 2-4 resolution stages (0.5→1.0, 0.33→0.66→1.0, 0.25→0.5→0.75→1.0). Starts cheap at low-res and upsamples when needed. Steps are placed automatically from Tolerance + A/beta.
"""

from __future__ import annotations

from speed_scripts.nodes_common import build_config_and_run

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
    progressive-resolution diffusion chain.

    """

    DESCRIPTION = (
        "Automatic SPEED sampler — just pick how many stages (2, 3 or 4) and go. "
        "It starts the video at low resolution (cheap), then automatically upsamples "
        "to full resolution when the detail actually matters. Set Tolerance (1% = 0.01) "
        "to allow a little blur for speed, or lower for quality. Uses baked A/beta; "
        "re-calibrate with the Harvest node if you change checkpoint. Audio is passed "
        "through at full-res."
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
        # Backwards compat: old workflows saved preset=half_then_full etc — map to stages
        if "preset" in kwargs:
            preset = kwargs.pop("preset")
            stages = PRESET_TO_STAGES.get(preset, stages)
        # Also accept stages as string from old INT widget serialization
        try:
            stages = int(stages)
        except Exception:
            stages = 3
        stages = max(2, min(4, stages))
        scales = STAGES_TO_SCALES[stages]
        # Dummy steps — validated then overridden by delta_custom power-spectrum thresholds in h3_runtime
        transition_steps = tuple(range(1, len(scales)))
        # Automatic node is delta_custom only — steps auto-computed from A/beta + Tolerance via thr formula
        config_mode = "delta_custom"
        return build_config_and_run(
            noise, guider, sigmas, latent_image,
            scales=scales,
            transition_steps=transition_steps,
            transition_mode=config_mode,
            noise_policy=noise_policy,
            delta=delta,
            noise_amplitude=noise_amplitude,
            noise_decay_exponent=noise_decay_exponent,
            seed_offset=seed_offset,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler"]
