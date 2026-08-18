"""Radial DCT power-spectrum harvesting and power-law fitting for MiniMax-H3.

Ported from the Lab's `speed_lab/tools.py`. Fits P = A * |omega|^(-beta) from
the *residual* noise field (x - x0) captured across a denoising pass. Used to
calibrate delta-optimal SPEED transition thresholds.

The finalized payload includes a `recommended_config` section that runs the
fitted (A, beta) through the delta-optimal activation-time formula for every
scale preset, producing ready-to-use transition_steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .spectral import dct2
from .h3_runtime import power_spectrum, activation_time
from .config import SCALE_PRESETS

try:  # package-relative when imported inside the repo
    from ..h3_logging import get_logger
except ImportError:  # root-relative when the repo root is on sys.path
    from h3_logging import get_logger

log = get_logger("Harvest")


def radial_power_spectrum(video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean 2D-DCT power of a video latent [B, C, T, H, W], binned radially.

    Returns (frequencies, mean_power) as numpy arrays.
    """
    H, W = video.shape[-2], video.shape[-1]
    coeffs = dct2(video.float())  # [B, C, T, H, W]
    power = coeffs.abs() ** 2
    power = power.mean(dim=(0, 1, 2))  # [H, W]

    cx, cy = W // 2, H // 2
    yy, xx = np.mgrid[0:H, 0:W]
    radial = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(int)
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


def _fit_health(fit: dict) -> str:
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


def recommend_configs(
    A: float, beta: float, sigmas, latent_h: int, latent_w: int,
    delta: float = 0.01,
) -> dict:
    """Compute delta-optimal transition_steps for every scale preset.

    Returns {preset_name: {scales, transition_steps, activation_thresholds}}.
    Uses the same activation_time / find_first_step_below formula as the
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
            p = power_spectrum(omega_i, A, beta)
            t_star = activation_time(p, delta)
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


@dataclass
class HarvestCallback:
    """A guider.sample callback that captures residual (x - x0) power per sigma."""

    sigmas: torch.Tensor
    every: int = 1
    profiles: list = field(default_factory=list)

    def __call__(self, step, x0, x, total_steps):
        if step % max(1, self.every) != 0 and step != total_steps - 1:
            return
        try:
            x_streams = list(x.unbind())
            x0_streams = list(x0.unbind())
        except AttributeError:
            log.warning("harvest callback step=%s: x/x0 are not NestedTensors "
                        "(unbind failed) — is the guider producing flat tensors?", step)
            return
        if len(x_streams) != 2 or len(x0_streams) != 2:
            log.warning("harvest callback step=%s: expected 2 streams, got %d/%d",
                        step, len(x_streams), len(x0_streams))
            return
        x_video, _ = x_streams
        x0_video, _ = x0_streams
        if x_video.ndim != 5 or x0_video.ndim != 5:
            log.warning("harvest callback step=%s: video ndim %d/%d (expected 5)",
                        step, x_video.ndim, x0_video.ndim)
            return
        residual = x_video - x0_video
        sigma = float(self.sigmas[min(step, len(self.sigmas) - 1)])
        freqs, profile = radial_power_spectrum(residual)
        self.profiles.append((sigma, freqs, profile))
        log.debug("harvest capture: step=%s/%s sigma=%.4f bins=%d mean_power=%.3e",
                  step, total_steps, sigma, len(freqs), float(np.asarray(profile).mean()))

    def finalize(
        self,
        omega_min: float = 0.5,
        *,
        latent_h: int | None = None,
        latent_w: int | None = None,
        delta: float = 0.01,
        fit_mode: str = "first",
    ) -> dict:
        """Return the fitted spectrum plus delta-optimal recommendations."""
        if not self.profiles:
            log.error("harvest finalize: no latents captured — the callback NEVER fired. "
                      "Check that guider.sample actually calls callbacks (ComfyUI "
                      ">= 0.30 does); if using BasicGuider, inspect guider.sample's "
                      "signature. Also check that the input latent is a real "
                      "NestedTensor from MiniMaxH3ImageToVideo (not a flat LATENT).")
            raise RuntimeError("no latents captured (callback never fired)")

        per_sigma_fits = []
        for sigma, freqs, power in self.profiles:
            vel = np.asarray(power) / max(float(sigma) ** 2, 1e-12)
            if (freqs >= omega_min).sum() >= 3:
                per_sigma_fits.append(
                    {"sigma": float(sigma), **fit_power_law(freqs, vel, omega_min)}
                )

        raw_profiles = [(p[1], np.asarray(p[2])) for p in self.profiles]
        if fit_mode == "first":
            if not per_sigma_fits:
                raise RuntimeError("no sigma level has enough bins to fit")
            fit = per_sigma_fits[0]
        elif fit_mode == "per_sigma":
            candidates = [f for f in per_sigma_fits if f["sigma"] < 0.99]
            fit = max(candidates or per_sigma_fits, key=lambda f: f["r_squared"])
        elif fit_mode == "pooled":
            all_freqs = np.concatenate([p[0] for p in raw_profiles])
            all_power = np.concatenate([p[1] for p in raw_profiles])
            fit = fit_power_law(all_freqs, all_power, omega_min)
        else:
            raise ValueError(f"unknown fit_mode {fit_mode!r}")

        result: dict = {
            "fit_mode": fit_mode,
            "sigma_levels": [float(p[0]) for p in self.profiles],
            "sigma_fits": per_sigma_fits,
            "overall_fit_A": fit["A"],
            "overall_fit_beta": fit["beta"],
            "overall_fit_r2": fit["r_squared"],
            "n_frequency_bins": fit["n_bins"],
            "fit_health": _fit_health(fit),
            "per_sigma_profiles": [
                {"sigma": float(p[0]), "freqs": p[1].tolist(),
                 "power": np.asarray(p[2]).tolist()}
                for p in self.profiles
            ],
        }
        if latent_h is not None and latent_w is not None:
            result["recommended_config"] = recommend_configs(
                fit["A"], fit["beta"], self.sigmas,
                latent_h, latent_w, delta=delta,
            )
        return result


__all__ = [
    "radial_power_spectrum", "fit_power_law", "_fit_health",
    "recommend_configs", "HarvestCallback",
]
