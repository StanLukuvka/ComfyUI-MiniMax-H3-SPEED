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
from minimax_h3_speed.h3_runtime import run_repeated_stage_calls, _unpack_tensor

# Human-readable dropdown labels (what the user sees) mapped back to the
# internal codes the config layer speaks. Labels stay space-separated; the
# raw underscore codes never reach the UI.
PRESET_LABELS = {
    "Half then Full": "half_then_full",
    "Quarter, Half, Full": "quarter_half_full",
    "Quarter, Half, 3Q, Full": "quarter_half_3q_full",
    "Aggressive": "aggressive",
    "Three Quarter then Full": "three_quarter_then_full",
}

TRANSITION_MODE_LABELS = {
    "Manual Step": "manual_step",
    "Manual Sigma": "manual_sigma",
    "Delta Custom": "delta_custom",
}

NOISE_POLICY_LABELS = {
    "Direct Coarse": "direct_coarse",
    "Coupled Full Grid": "coupled_full_grid",
}


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
                "explicit_preset": (list(PRESET_LABELS.keys()),),
                "transition_mode": (list(TRANSITION_MODE_LABELS.keys()),),
                "noise_policy": (list(NOISE_POLICY_LABELS.keys()), {"default": "Direct Coarse"}),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
                "power_A": ("FLOAT", {"default": 150.0, "min": 0.0, "max": 1e6}),
                "power_beta": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0}),
                "seed_offset": ("INT", {"default": 10000, "min": 0, "max": 2**31 - 1}),
            },
        }

    def sample(self, noise, guider, sigmas, latent_image, explicit_preset,
               transition_mode, noise_policy="Direct Coarse",
               delta=0.01, power_A=150.0, power_beta=2.0,
               seed_offset=10000):
        # Translate UI labels back to the internal codes used everywhere else
        # (config, runtime, tests). Unknown labels fail loudly rather than
        # silently falling back.
        preset_code = PRESET_LABELS[explicit_preset]
        transition_code = TRANSITION_MODE_LABELS[transition_mode]
        noise_code = NOISE_POLICY_LABELS.get(noise_policy, "direct_coarse")

        # Use the calibrated transition steps from the preset config.
        # These are tuned per-preset (e.g. 2_stage_half transitions at step 5
        # out of 20, leaving 15 steps for full-res detail refinement).
        scales = SCALE_PRESETS[preset_code]
        transition_steps = DEFAULT_TRANSITION_STEPS[preset_code]
        n_stages = len(scales)
        n_sigmas = len(sigmas)
        # Need at least 2 sigmas per stage, plus enough room for transition steps.
        # max(transition_steps) is the last boundary index, so we need that + 1
        # to cover the final stage's sigma slice.
        min_required = max(n_stages * 2, max(transition_steps) + 1)
        if n_sigmas < min_required:
            raise ValueError(
                f"sigma schedule too short: got {n_sigmas} sigmas, need "
                f"at least {min_required} for preset '{preset_code}' with "
                f"transition steps {transition_steps}. "
                f"Increase steps to >= {min_required - 1}."
            )
        # Delta-custom mode gets (A, β) from a prior SigmaHarvest run; explicit
        # mode uses the manually tuned transition_steps in the config.
        full_video, _ = _unpack_tensor(latent_image.get("samples"))
        # Map node-facing transition_mode values to config-internal vocabulary.
        # "manual_step" and "manual_sigma" both produce explicit transition_steps
        # (resolved from the preset); only "delta_custom" uses power-spectrum
        # thresholds.
        MODE_TO_CONFIG = {"manual_step": "explicit",
                          "manual_sigma": "explicit",
                          "delta_custom": "delta_custom"}
        config_mode = MODE_TO_CONFIG.get(transition_code, "explicit")
        config = SpeedConfig(
            scales=scales,
            transition_steps=transition_steps,
            transition_mode=config_mode,
            noise_policy=noise_code,
            delta=float(delta),
            power_A=float(power_A),
            power_beta=float(power_beta),
            transition_seed_offset=int(seed_offset),
            full_latent_h=int(full_video.shape[-2]),
            full_latent_w=int(full_video.shape[-1]),
        )

        # Run the multi-stage SPEED diffusion chain. Audio is carried through
        # unchanged; the final-stage x0 (denoised) is surfaced as denoised_output.
        out, denoised = run_repeated_stage_calls(
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

        return (out, denoised)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSampler": MiniMaxH3SPEEDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSampler": "MiniMax H3 SPEED — Sampler"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "MiniMaxH3SPEEDSampler"]
