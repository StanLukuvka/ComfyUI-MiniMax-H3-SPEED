"""Shared INPUT_TYPES pieces for the Advanced and KSampler-style SPEED nodes.

The KSampler base inputs match ComfyUI's ``nodes.py`` KSampler. Saved
workflows store widget values by position, so each node's key order is part
of the public contract.
"""

from __future__ import annotations

from typing import Any

import comfy.samplers

from speed_scripts.config import RATIO_MODES

# KSampler base inputs, identical to ComfyUI's nodes.py KSampler (same order,
# defaults, ranges and tooltips). The KSampler-style SPEED nodes append
# their mode's SPEED widgets after these.
KSAMPLER_BASE_INPUTS: dict[str, tuple[Any, dict[str, Any]]] = {
    "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
    "seed": ("INT", {
        "default": 0,
        "min": 0,
        "max": 0xffffffffffffffff,
        "control_after_generate": True,
        "tooltip": "The random seed used for creating the noise.",
    }),
    "steps": ("INT", {
        "default": 20,
        "min": 1,
        "max": 10000,
        "tooltip": "The number of steps used in the denoising process.",
    }),
    "cfg": ("FLOAT", {
        "default": 8.0,
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "round": 0.01,
        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
    }),
    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
        "tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output.",
    }),
    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
        "tooltip": "The scheduler controls how noise is gradually removed to form the image.",
    }),
    "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
    "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
    "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
    "denoise": ("FLOAT", {
        "default": 1.0,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling.",
    }),
}

# Advanced base inputs (SamplerCustomAdvanced style). The sampler tooltip
# differs between the Automatic and Manual nodes, so the base holds the four
# shared inputs and each node inserts its own SAMPLER entry.
ADVANCED_BASE_INPUTS: dict[str, tuple[str, dict[str, Any]]] = {
    "noise": (
        "NOISE",
        {"tooltip": "Noise source, same as SamplerCustomAdvanced (e.g. RandomNoise). Its seed also seeds the sampler."},
    ),
    "guider": (
        "GUIDER",
        {"tooltip": "Guider, same as SamplerCustomAdvanced (e.g. CFGGuider, BasicGuider)."},
    ),
    "sigmas": (
        "SIGMAS",
        {"tooltip": "Sigma schedule, same as SamplerCustomAdvanced. Stage boundaries are placed on this schedule."},
    ),
    "latent_image": (
        "LATENT",
        {"tooltip": "Latent to sample, same as SamplerCustomAdvanced. The final stage runs at this full size."},
    ),
}
SAMPLER_INPUT_AUTOMATIC: tuple[str, dict[str, Any]] = (
    "SAMPLER",
    {"tooltip": "Any ComfyUI SAMPLER (e.g. from KSamplerSelect). Run Sigma Harvest with the same sampler; the shipped H3 reference values are Euler-only."},
)
SAMPLER_INPUT_MANUAL: tuple[str, dict[str, Any]] = (
    "SAMPLER",
    {"tooltip": "Any ComfyUI SAMPLER (e.g. from KSamplerSelect). Any sampler works; Euler is the one SPEED was measured with."},
)

# Automatic mode widgets.
SPEED_AUTOMATIC_WIDGETS: dict[str, tuple[Any, dict[str, Any]]] = {
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
            "tooltip": "direct_coarse: start on coarse-size noise and fill each new frequency band from seeded noise at every transition. coupled_full_grid: take all bands from one full-resolution noise field drawn like SamplerCustomAdvanced, so it matches a plain run at the same seed. On H3 video it saved only 5-10%, against 15-30% for direct_coarse.",
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
            "default": 0.0,
            "min": 0.0,
            "max": 1e6,
            "step": 0.0001,
            "round": 0.0001,
            "tooltip": "A in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule. There is no default: 0 means not set and stops with an error.",
        },
    ),
    "noise_decay_exponent": (
        "FLOAT",
        {
            "default": 0.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.0001,
            "round": 0.0001,
            "tooltip": "beta in the fitted power spectrum P = A*|w|^-beta. Paste the value from Sigma Harvest for your model, sampler and schedule. There is no default: 0 means not set and stops with an error.",
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
}

# Manual mode widgets.
SPEED_MANUAL_WIDGETS: dict[str, tuple[Any, dict[str, Any]]] = {
    "noise_policy": (
        ["direct_coarse", "coupled_full_grid"],
        {
            "default": "direct_coarse",
            "tooltip": "direct_coarse: start on coarse-size noise and fill each new frequency band from seeded noise at every transition. coupled_full_grid: take all bands from one full-resolution noise field drawn like SamplerCustomAdvanced, so it matches a plain run at the same seed. On H3 video it saved only 5-10%, against 15-30% for direct_coarse.",
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
    "ratio_mode": (
        list(RATIO_MODES),
        {
            "default": "steps",
            "tooltip": "How transition goals are read. steps: whole step indices into the sigma schedule. ratio: 0-1 fractions of the schedule, placed at round(goal * total steps).",
        },
    ),
    "transition_goal_1": (
        "FLOAT",
        {
            "default": 3,
            "min": 0,
            "max": 1000,
            "tooltip": "Where stage 1 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
        },
    ),
    "transition_resolution_1": (
        "FLOAT",
        {
            "default": 0.25,
            "min": 0,
            "max": 1,
            "tooltip": "Stage 1 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
        },
    ),
    "transition_goal_2": (
        "FLOAT",
        {
            "default": 5,
            "min": 0,
            "max": 1000,
            "tooltip": "Where stage 2 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
        },
    ),
    "transition_resolution_2": (
        "FLOAT",
        {
            "default": 0.5,
            "min": 0,
            "max": 1,
            "tooltip": "Stage 2 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
        },
    ),
    "transition_goal_3": (
        "FLOAT",
        {
            "default": 8,
            "min": 0,
            "max": 1000,
            "tooltip": "Where stage 3 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
        },
    ),
    "transition_resolution_3": (
        "FLOAT",
        {
            "default": 0.75,
            "min": 0,
            "max": 1,
            "tooltip": "Stage 3 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
        },
    ),
    "transition_goal_4": (
        "FLOAT",
        {
            "default": 15,
            "min": 0,
            "max": 1000,
            "tooltip": "Where stage 4 ends. In steps mode, a whole global step index; in ratio mode, a 0-1 fraction of the schedule. 0 disables this stage. The last active stage runs to the end of the schedule, so its goal only needs to be non-zero.",
        },
    ),
    "transition_resolution_4": (
        "FLOAT",
        {
            "default": 1.0,
            "min": 0,
            "max": 1,
            "tooltip": "Stage 4 scale relative to full size (0.5 = half width and height). 0 disables this stage. The last active stage must be 1.0.",
        },
    ),
}

# Harvest mode widgets.
SPEED_HARVEST_WIDGETS: dict[str, tuple[Any, dict[str, Any]]] = {
    "Tolerance (Delta)": (
        "FLOAT",
        {
            "default": 0.005,
            "min": 1e-4,
            "max": 0.5,
            "step": 0.001,
            "tooltip": "The Delta the report computes transition boundaries for. It is written into the calibration and the paste line; the fit itself does not depend on it.",
        },
    ),
}
