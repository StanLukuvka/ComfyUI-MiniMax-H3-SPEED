# FLOW-PRODUCED — Implementation Plan — Continuous SPEED Sigma Harvester.md §17-24, 26, 30, 35, 42 (commit 3) — flow-produced, do not hand-edit
"""Torch-native online spectral analysis for continuous SPEED sigma harvesting.

Per-step companion to :mod:`speed_scripts.harvest`: measures the radial DCT
power spectrum of a video tensor and fits ``P = A * |omega|^(-beta)`` without
leaving the torch device, so a per-callback collector never pays for GPU
sync or a ``.cpu().numpy()`` transfer. Also provides direct point/band
sampling of a radial profile, a log-space EMA for boundary-power smoothing,
and JSON-safe fit records (failed fits become nulls, never NaN/Infinity).

Pure functions only — no node and no runtime wiring lives here.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from .harvest import classify_fit_quality
from .spectral import dct2

# Plan §21: keep the lowest fitted bin just above DC. omega_min = 0.5 selects
# every integer radial frequency >= 1 while excluding the DC bin (omega = 0),
# and is never changed between stages.
DEFAULT_OMEGA_MIN = 0.5

# Plan §35 smoothing alpha for per-stage log-space EMAs.
DEFAULT_SMOOTHING_ALPHA = 0.25


class _RadialBinsKey(NamedTuple):
    height: int
    width: int
    device_type: str
    device_index: int | None


@lru_cache(maxsize=64)
def _cached_radial_bins(
    height: int, width: int, device_type: str, device_index: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    radial = torch.round(torch.sqrt(xx * xx + yy * yy)).to(torch.int64)
    max_r = int(radial.max().item())
    indices = radial.flatten()
    counts = torch.bincount(indices, minlength=max_r + 1)
    return indices, counts


def _build_radial_bins(height: int, width: int, device: torch.device):
    return _cached_radial_bins(height, width, device.type, device.index)


def radial_dct_power_torch(video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean 2D-DCT power of a video [B, C, T, H, W], binned radially on device.

    Returns ``(freqs, profile)`` as 1-D float32 tensors on the input's device:
    ``freqs`` are the non-empty integer radial frequencies in ascending order,
    ``profile`` the mean squared DCT coefficient at each frequency. No element
    of the computation touches NumPy or the CPU.
    """
    if video.ndim != 5:
        raise ValueError(
            f"radial_dct_power_torch expects a [B, C, T, H, W] video; got ndim={video.ndim}"
        )
    height, width = video.shape[-2], video.shape[-1]
    coeffs = dct2(video.detach().float())
    power = coeffs.square().mean(dim=(0, 1, 2))  # [H, W]

    indices, counts = _build_radial_bins(height, width, video.device)
    sums = torch.bincount(indices, weights=power.flatten(), minlength=counts.numel())
    valid = counts > 0
    freqs = torch.arange(counts.numel(), device=power.device, dtype=torch.float32)[valid]
    profile = sums[valid] / counts[valid]
    return freqs, profile


def fit_power_law_torch(
    freqs: torch.Tensor,
    profile: torch.Tensor,
    omega_min: float = DEFAULT_OMEGA_MIN,
    omega_max: float | None = None,
) -> dict[str, Any]:
    """Fit ``P = A * omega^(-beta)`` by OLS in log-log space, torch-native.

    Mirrors :func:`speed_scripts.harvest.fit_power_law`: same mask convention
    (``omega >= omega_min`` and ``P > 0``), same R-squared, and the same
    health classification thresholds via :func:`classify_fit_quality`. The
    regression itself runs in float64 so results line up with the NumPy
    fitter's float64 polyfit. ``omega_max`` caps the fit range from above
    (plan §21: ``min(H_stage, W_stage) / 2``); the default keeps every
    non-empty bin. Returns a dict with ``A``, ``beta``, ``r_squared``,
    ``n_bins``, ``status``, and ``health``; a fit with too few bins or any
    non-finite value gets ``status="fit_failed"`` with ``None`` numerics.
    """
    failed: dict[str, Any] = {
        "status": "fit_failed",
        "A": None,
        "beta": None,
        "r_squared": None,
        "n_bins": 0,
    }
    freqs_f = freqs.detach().double()
    profile_f = profile.detach().double()
    if freqs_f.numel() == 0 or freqs_f.numel() != profile_f.numel():
        return failed

    mask = (freqs_f >= omega_min) & (profile_f > 0)
    if omega_max is not None:
        mask = mask & (freqs_f <= omega_max)
    x = torch.log(freqs_f[mask])
    y = torch.log(profile_f[mask])
    if x.numel() < 3:
        failed["n_bins"] = int(x.numel())
        return failed

    x_mean = x.mean()
    y_mean = y.mean()
    slope = ((x - x_mean) * (y - y_mean)).mean() / ((x - x_mean) ** 2).mean()
    intercept = y_mean - slope * x_mean

    pred = intercept + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fit = {
        "A": math.exp(float(intercept)),
        "beta": float(-slope),
        "r_squared": r2,
        "n_bins": int(x.numel()),
    }
    if not all(math.isfinite(fit[key]) for key in ("A", "beta", "r_squared")):
        failed["n_bins"] = fit["n_bins"]
        return failed
    fit["status"] = "ok"
    fit["health"] = classify_fit_quality(fit)
    return fit


def sample_radial_power(freqs: torch.Tensor, profile: torch.Tensor, omega: float) -> float:
    """Point sample of a radial profile at ``omega`` by linear interpolation
    between the neighboring bins (plan §30). Values below or above the measured
    range clamp to the first or last bin instead of extrapolating.
    """
    if freqs.numel() == 0:
        raise ValueError("cannot sample an empty radial profile")
    x = freqs.detach().float()
    y = profile.detach().float()
    order = torch.argsort(x)
    x = x[order]
    y = y[order]
    omega = float(omega)
    if omega <= float(x[0]):
        return float(y[0])
    if omega >= float(x[-1]):
        return float(y[-1])
    upper = int(torch.clamp(torch.searchsorted(x, torch.tensor(omega)), 1, x.numel() - 1))
    x0 = float(x[upper - 1])
    x1 = float(x[upper])
    y0 = float(y[upper - 1])
    y1 = float(y[upper])
    if x1 == x0:
        return y0
    t = (omega - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def sample_radial_band_power(
    freqs: torch.Tensor, profile: torch.Tensor, omega: float, half_width: float = 1.0
) -> float:
    """Narrow-band mean of a radial profile over ``[omega - half_width, omega + half_width]``.

    Falls back to the interpolated point sample when no bin falls inside the
    band, so a caller never gets an error for a band that is empty at the
    current stage resolution.
    """
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    x = freqs.detach().float()
    y = profile.detach().float()
    mask = (x >= omega - half_width) & (x <= omega + half_width)
    if not bool(mask.any()):
        return sample_radial_power(freqs, profile, omega)
    return float(y[mask].mean())


def update_log_ema(
    previous_ema: float | None, raw: float, alpha: float = DEFAULT_SMOOTHING_ALPHA
) -> float:
    """One log-space EMA step: ``log_ema = (1-alpha)*log_prev + alpha*log(raw)``.

    The first valid measurement seeds the EMA (plan §36 — never seed from the
    baked calibration A/beta). A non-positive ``raw`` carries no log-space
    information and leaves the previous value unchanged. Returns the EMA in
    linear power units.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if raw <= 0:
        return previous_ema if previous_ema is not None else float("nan")
    if previous_ema is None:
        return float(raw)
    if previous_ema <= 0:
        raise ValueError("previous_ema must be positive once seeded")
    log_ema = (1.0 - alpha) * math.log(previous_ema) + alpha * math.log(raw)
    return math.exp(log_ema)


def finite_or_none(value: Any) -> float | None:
    """Convert to a finite float, or None for None/NaN/±Infinity (plan §42)."""
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def build_fit_record(fit: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-safe record for one spectral fit (plan §42).

    A missing or failed fit becomes ``{"status": "fit_failed", "A": null,
    "beta": null, "r_squared": null}``; a successful fit carries finite
    floats plus ``n_bins`` and ``health``. The result always survives
    ``json.dumps(..., allow_nan=False)``.
    """
    failed: dict[str, Any] = {
        "status": "fit_failed",
        "A": None,
        "beta": None,
        "r_squared": None,
    }
    if fit is None:
        return failed
    record = {
        "status": "ok",
        "A": finite_or_none(fit["A"]),
        "beta": finite_or_none(fit["beta"]),
        "r_squared": finite_or_none(fit["r_squared"]),
        "n_bins": int(fit["n_bins"]),
    }
    if any(record[key] is None for key in ("A", "beta", "r_squared")):
        return failed
    record["health"] = str(fit["health"])
    return record


def dumps_strict(document: dict[str, Any]) -> str:
    """Serialize a harvest document with NaN/Infinity rejected (plan §42)."""
    return json.dumps(document, allow_nan=False)


__all__ = [
    "DEFAULT_OMEGA_MIN",
    "DEFAULT_SMOOTHING_ALPHA",
    "radial_dct_power_torch",
    "fit_power_law_torch",
    "sample_radial_power",
    "sample_radial_band_power",
    "update_log_ema",
    "finite_or_none",
    "build_fit_record",
    "dumps_strict",
]
