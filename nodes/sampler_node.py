"""ComfyUI node definitions for MiniMax-H3 SPEED.

Replaced Ksampler + SamplerCustomAdvanced. SPEED is only developed with euler and SamplerCustomAdvanced only supports one iteration of latent settings at a time.
TLDR: Default sampler didnt expect you to change the resolution mid flight cause why would it?

SPEED basics: denoise video as low res at low steps as no value is given from making fancy noise at high resolution.
Then we step increase resolution depending on preset and continue. 
Audio stays unchanged and done at full resolution. 
"""

from __future__ import annotations

from speed_scripts.config import SCALE_PRESETS, DEFAULT_TRANSITION_STEPS
from speed_scripts.nodes_common import build_config_and_run

class MiniMaxH3SPEEDSampler:
    """SPEED progressive-resolution diffusion for MiniMax-H3's packed latent.

    Drop-in replacement for the standard KSAMPLER + SamplerCustomAdvanced
    pair. Takes (noise, guider, sigmas, latent_image) and runs a multi-stage
    progressive-resolution diffusion chain.

    """

    DESCRIPTION = (
        "SPEED progressive-resolution diffusion for MiniMax-H3 packed latents. "
        "Runs each stage as its own guider.sample() call. low res first until we reach point where information generated matters. then increase resolution"
        "first preset scale, then DCT-expand + scale-ratio-aligned boundary sigma,"
        "then full-res pass. Audio is carried through unchanged."
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
                "preset": (list(SCALE_PRESETS.keys()),),
                "transition_mode": (["manual_step", "manual_sigma", "delta_custom"],),
                "noise_policy": (["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"}),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "noise_amplitude": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 1e6}),
                "noise_decay_exponent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, preset,
               transition_mode, noise_policy="direct_coarse",
               delta=0.01, noise_amplitude=150.0, noise_decay_exponent=2.0,
               seed_offset=10000):
        scales = SCALE_PRESETS[preset]
        transition_steps = DEFAULT_TRANSITION_STEPS[preset]
        # Delta-custom mode gets (A, beta) from a prior SigmaHarvest run; explicit
        # mode uses the manually tuned transition_steps in the config.
        # Map node-facing transition_mode values to config-internal vocabulary.
        # "manual_step" and "manual_sigma" both produce explicit transition_steps
        # (resolved from the preset); only "delta_custom" uses power-spectrum
        # thresholds. (The deleted Schedule node used to emit the same three values;
        # its vocabulary was folded into this mapping when it was pruned.)
        transition_mode_map = {"manual_step": "explicit",
                          "manual_sigma": "explicit",
                          "delta_custom": "delta_custom"}
        config_mode = transition_mode_map.get(transition_mode, "explicit")
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
