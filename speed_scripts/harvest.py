"""Radial DCT power-spectrum analysis for MiniMax-H3 SPEED calibration.

Fits P = A * |omega|^(-beta) from a residual noise field (x - x0), and turns the
fitted (A, beta) into delta-optimal SPEED transition_steps for every scale preset.

This module is the *consumer-side* math. The actual capture of the residual field
must happen on a NATIVE single-resolution sampler pass (not inside the SPEED sampler,
whose multi-stage sigma splicing makes per-step sigma labeling incorrect). The
harvested JSON is then fed to the MiniMaxH3HarvestToConfig node for a human-readable
report.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .spectral import dct2
from .h3_runtime import power_at_frequency, activation_threshold
from .config import SCALE_PRESETS


def radial_dct_power(video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean 2D-DCT power of a video latent [B, C, T, H, W], binned radially.

    Returns (frequencies, mean_power) as numpy arrays.
    """
    H, W = video.shape[-2], video.shape[-1]
    coeffs = dct2(video.float())  # [B, C, T, H, W]
    power = coeffs.abs() ** 2
    power = power.mean(dim=(0, 1, 2))  # [H, W]

    cx, rdp_cy = 0, 0
    yy, rdp_xx = np.mgrid[0:H, 0:W]
    radial = np.round(np.sqrt((rdp_xx - cx) ** 2 + (yy - rdp_cy) ** 2)).astype(int)
    max_r = radial.max()
    counts = np.bincount(radial.ravel(), minlength=max_r + 1)
    sums = np.bincount(radial.ravel(), weights=power.cpu().numpy().ravel(),
                       minlength=max_r + 1)
    valid = counts > 0
    freqs = np.arange(max_r + 1)[valid]
    profile = (sums / np.maximum(counts, 1))[valid]
    return freqs, profile


def fit_power_law(freqs: np.ndarray, profile: np.ndarray,
                  omega_min: float = 0.5) -> dict:
    """Fit P = A * omega^(-beta) on log-log. Returns {A, beta, r_squared, n_bins}."""
    mask = (freqs >= omega_min) & (profile > 0)
    fpl_x = np.log(freqs[mask])
    fpl_y = np.log(profile[mask])
    if len(fpl_x) < 3:
        raise ValueError("not enough frequency bins to fit power law")
    slope, intercept = np.polyfit(fpl_x, fpl_y, 1)
    beta = -slope
    A = math.exp(intercept)
    pred = intercept + slope * fpl_x
    ss_res = float(((fpl_y - pred) ** 2).sum())
    ss_tot = float(((fpl_y - fpl_y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"A": A, "beta": beta, "r_squared": r2, "n_bins": len(fpl_x)}


def classify_fit_quality(fit: dict) -> str:
    """Classify a spectral fit so downstream consumers can warn on bad ones."""
    a, beta, r2 = fit["A"], fit["beta"], fit["r_squared"]
    if a != a or beta != beta or r2 != r2:  # any nan
        return "invalid"
    if beta > 0 and r2 >= 0.7:
        return "good"
    if beta > 0 and r2 >= 0.4:
        return "fair"
    if beta > 0:
        return "weak"
    return "suspect"  # beta <= 0: power not decaying


def recommend_transition_steps(
    A: float, beta: float, sigmas, latent_h: int, latent_w: int,
    delta: float = 0.01,
) -> dict:
    """Compute delta-optimal transition_steps for every scale preset.

    Returns {preset_name: {scales, transition_steps, activation_thresholds}}.
    Uses the same activation_threshold / find_first_step_below formula as the
    runtime so the numbers match a delta_custom run.
    """
    sigmas_list = [float(s) for s in sigmas]
    n_sigmas = len(sigmas_list) - 1
    omega_max = min(latent_h, latent_w) / 2.0
    presets = {}
    for name, scales in SCALE_PRESETS.items():
        steps = []
        thresholds = []
        for i in range(len(scales) - 1):
            omega_i = scales[i] * omega_max
            rts_p = power_at_frequency(omega_i, A, beta)
            t_star = activation_threshold(rts_p, delta)
            thresholds.append(t_star)
            step = n_sigmas  # fallback
            for j in range(n_sigmas):
                if sigmas_list[j] <= t_star:
                    step = j
                    break
            steps.append(step)
        presets[name] = {
            "scales": list(scales),
            "transition_steps": steps,
            "activation_thresholds": thresholds,
        }
    return presets


__all__ = [
    "radial_dct_power", "fit_power_law", "classify_fit_quality",
    "recommend_transition_steps",
]
