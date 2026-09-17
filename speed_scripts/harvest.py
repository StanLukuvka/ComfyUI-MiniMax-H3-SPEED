"""Radial DCT power-spectrum analysis for MiniMax-H3 SPEED calibration.

Fits ``P = A * |omega|^(-beta)`` from a residual noise field (``x - x0``).
Residual capture itself happens on a native single-resolution sampler pass;
this module only owns the pure spectral analysis and fit-quality helpers.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .spectral import dct2


def radial_dct_power(video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean 2D-DCT power of a video latent [B, C, T, H, W], binned radially."""
    H, W = video.shape[-2], video.shape[-1]
    coeffs = dct2(video.float())
    power = coeffs.abs() ** 2
    power = power.mean(dim=(0, 1, 2))

    yy, xx = np.mgrid[0:H, 0:W]
    radial = np.round(np.sqrt(xx ** 2 + yy ** 2)).astype(int)
    max_r = radial.max()
    counts = np.bincount(radial.ravel(), minlength=max_r + 1)
    sums = np.bincount(
        radial.ravel(),
        weights=power.cpu().numpy().ravel(),
        minlength=max_r + 1,
    )
    valid = counts > 0
    freqs = np.arange(max_r + 1)[valid]
    profile = (sums / np.maximum(counts, 1))[valid]
    return freqs, profile


def fit_power_law(
    freqs: np.ndarray,
    profile: np.ndarray,
    omega_min: float = 0.5,
) -> dict:
    """Fit ``P = A * omega^(-beta)`` on log-log coordinates."""
    mask = (freqs >= omega_min) & (profile > 0)
    x = np.log(freqs[mask])
    y = np.log(profile[mask])
    if len(x) < 3:
        raise ValueError("not enough frequency bins to fit power law")
    slope, intercept = np.polyfit(x, y, 1)
    beta = -slope
    A = math.exp(intercept)
    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"A": A, "beta": beta, "r_squared": r2, "n_bins": len(x)}


def classify_fit_quality(fit: dict) -> str:
    """Classify a spectral fit so downstream consumers can warn on bad ones."""
    a, beta, r2 = fit["A"], fit["beta"], fit["r_squared"]
    if a != a or beta != beta or r2 != r2:
        return "invalid"
    if beta > 0 and r2 >= 0.7:
        return "good"
    if beta > 0 and r2 >= 0.4:
        return "fair"
    if beta > 0:
        return "weak"
    return "suspect"


__all__ = ["radial_dct_power", "fit_power_law", "classify_fit_quality"]
