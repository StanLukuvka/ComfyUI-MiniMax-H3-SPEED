"""ComfyUI node definitions for MiniMax-H3 SPEED.

Replaced Ksampler + SamplerCustomAdvanced. SPEED is only developed with euler and SamplerCustomAdvanced only supports one iteration of latent settings at a time.
TLDR: Default sampler didnt expect you to change the resolution mid flight cause why would it?

SPEED basics: denoise video as low res at low steps as no value is given from making fancy noise at high resolution.
Then we step increase resolution depending on preset and continue. 
Audio stays unchanged and done at full resolution. 
"""

from __future__ import annotations

import comfy.samplers
from minimax_h3_speed.config import SCALE_PRESETS, DEFAULT_TRANSITION_STEPS, SpeedConfig
from minimax_h3_speed.h3_runtime import run_progressive_stages, _unpack_tensor

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
                "explicit_preset": (list(SCALE_PRESETS.keys()),),
                "transition_mode": (["explicit", "delta_custom"],),
                "noise_policy": (["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"}),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "power_A": ("FLOAT", {"default": 219.48, "min": 0.0, "max": 1e6}),
                "power_beta": ("FLOAT", {"default": 2.42, "min": 0.0, "max": 10.0}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, explicit_preset,
               transition_mode, noise_policy="direct_coarse",
               delta=0.01, power_A=219.48, power_beta=2.42,
               seed_offset=10000):
        # Use the calibrated transition steps from the preset config.
        # These are tuned per-preset (e.g. 2_stage_half transitions at step 5
        # out of 20, leaving 15 steps for full-res detail refinement).
        scales = SCALE_PRESETS[explicit_preset]
        transition_steps = DEFAULT_TRANSITION_STEPS[explicit_preset]
        n_stages = len(scales)
        n_sigmas = len(sigmas)
        # Need at least 2 sigmas per stage, plus enough room for transition steps.
        # max(transition_steps) is the last boundary index, so we need that + 1
        # to cover the final stage's sigma slice.
        min_required = max(n_stages * 2, max(transition_steps) + 1)
        if n_sigmas < min_required:
            raise ValueError(
                f"sigma schedule too short: got {n_sigmas} sigmas, need "
                f"at least {min_required} for preset '{explicit_preset}' with "
                f"transition steps {transition_steps}. "
                f"Increase steps to >= {min_required - 1}."
            )
        # Delta-custom mode gets (A, β) from a prior SigmaHarvest run; explicit
        # mode uses the manually tuned transition_steps in the config.
        full_video, _ = _unpack_tensor(latent_image.get("samples"))
        config = SpeedConfig(
            scales=scales,
            transition_steps=transition_steps,
            transition_mode=transition_mode,
            noise_policy=noise_policy,
            delta=float(delta),
            power_A=float(power_A),
            power_beta=float(power_beta),
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )
        

        return run_progressive_stages(
            noise,
            guider,
            sigmas,
            latent_image,
            config,
            # NOTE: SPEED's kappa alignment and DCT-boundary semantics are
            # calibrated for Euler. Supporting other samplers requires calibration and changing the underlying math for kappa alignment.
            # This has been left as an exercise for the reader.
            sampler=comfy.samplers.sampler_object("euler"),
            nested_type=comfy.nested_tensor.NestedTensor,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            output_device=None,
        )


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler"]
