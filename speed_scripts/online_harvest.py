"""Torch-native online spectral analysis for continuous SPEED sigma harvesting.

Per-step companion to :mod:`speed_scripts.harvest`: measures the radial DCT
power spectrum of a video tensor and fits ``P = A * |omega|^(-beta)`` without
leaving the torch device, so a per-callback collector never pays for GPU
sync or a ``.cpu().numpy()`` transfer. Also provides direct point/band
sampling of a radial profile, a log-space EMA for boundary-power smoothing,
and JSON-safe fit records (failed fits become nulls, never NaN/Infinity).

Also hosts the `SpeedHarvestCollector` — the observer-side collector that turns
SPEED runtime events into per-step telemetry records. No ComfyUI node lives
here; the node wiring stays in nodes/.
"""

from __future__ import annotations

import json
import math
import time
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from .harvest import classify_fit_quality
from .h3_runtime import (
    _find_first_step_below as first_step_below,
    activation_threshold,
    power_at_frequency,
)
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


class SpeedHarvestCollector:
    """Observer-side collector: SPEED runtime events -> telemetry records.

    Implements the `SpeedRuntimeObserver` protocol (plan §9). The runtime
    decides *what is happening* (stage slices, boundaries, sigma values);
    this collector decides *what to measure*: per-callback absolute radial
    DCT power of the x0 video and (mode permitting) the residual x - x0,
    power-law fits, direct boundary-power estimates, live activation
    thresholds, log-space EMA smoothing, and the JSON document (plans
    §24-§41).

    Tensor lifetime (plan §26, non-negotiable): every callback tensor is
    received, reduced to Python scalars / small CPU lists, and dropped
    before `on_step` returns. No tensor is ever appended to the records.
    """

    def __init__(
        self,
        *,
        delta: float,
        noise_amplitude: float,
        noise_decay_exponent: float,
        measurement_mode: str = "both",
        analysis_stride: int = 1,
        smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA,
        boundary_band_half_width: float = 1.0,
        store_radial_profiles: bool = False,
        record_timing: bool = False,
    ):
        if measurement_mode not in ("both", "x0_only", "residual_only"):
            raise ValueError(f"unsupported measurement_mode: {measurement_mode}")
        if not 0.0 < smoothing_alpha < 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1)")
        if analysis_stride < 1:
            raise ValueError("analysis_stride must be >= 1")
        self.delta = float(delta)
        self.noise_amplitude = float(noise_amplitude)
        self.noise_decay_exponent = float(noise_decay_exponent)
        self.measurement_mode = measurement_mode
        self.analysis_stride = int(analysis_stride)
        self.smoothing_alpha = float(smoothing_alpha)
        self.boundary_band_half_width = float(boundary_band_half_width)
        self.store_radial_profiles = bool(store_radial_profiles)
        # Plan §58: analysis_ms is recorded only when explicitly requested —
        # honest timing needs a device sync, which itself changes performance.
        self.record_timing = bool(record_timing)

        self.records: list[dict[str, Any]] = []
        self.transitions: list[dict[str, Any]] = []
        self._static_scheduler: dict[str, Any] = {}
        self._analysis_block: dict[str, Any] = {}
        self._original_sigmas: list[float] = []
        self._scales: list[float] = []
        self._full_h = 0
        self._full_w = 0
        # (transition_index, omega) pairs this stage can still measure or
        # predict: the current boundary first, then future boundaries.
        self._future_omegas: list[tuple[int, float]] = []
        self._omega_boundary: float | None = None
        self._next_transition_index: int = 0
        # Per-stage log-space EMAs, x0 and residual namespaces separate
        # (plans §35, §37). Keyed by stage_index; each starts None and is
        # seeded from the stage's FIRST valid measurement — never from the
        # static calibration coefficients (plan §36).
        self._ema_x0: dict[int, float | None] = {}
        self._ema_residual: dict[int, float | None] = {}

    # ------------------------------------------------------------------
    # Observer protocol
    # ------------------------------------------------------------------
    def on_run_start(self, event) -> None:
        self._scales = [float(s) for s in event.scales]
        self._full_h = event.full_h
        self._full_w = event.full_w
        self._static_scheduler = {
            "stages": event.n_stages,
            "scales": self._scales,
            "delta": self.delta,
            "noise_amplitude": self.noise_amplitude,
            "noise_decay_exponent": self.noise_decay_exponent,
            "resolved_transition_steps": [int(s) for s in event.transition_steps],
            "original_sigmas": list(self._original_sigmas),
        }
        self._analysis_block = {
            "stride": self.analysis_stride,
            "smoothing_alpha": self.smoothing_alpha,
            "boundary_band_half_width": self.boundary_band_half_width,
            "fit_omega_min": DEFAULT_OMEGA_MIN,
            "fit_omega_max_policy": "half_min_current_stage",
            "measurement_mode": self.measurement_mode,
            "store_radial_profiles": self.store_radial_profiles,
        }
        self._arm_stage(0)

    def on_step(self, event, x0, x) -> None:
        if event.stage_local_step == 0:
            # First measured callback of a stage re-arms the per-stage EMAs.
            self._ema_x0.setdefault(event.stage_index, None)
            self._ema_residual.setdefault(event.stage_index, None)

        measured = event.callback_index % self.analysis_stride == 0
        record: dict[str, Any] = {
            "callback_index": event.callback_index,
            "global_schedule_index": event.global_schedule_index,
            "stage_index": event.stage_index,
            "stage_scale": float(event.stage_scale),
            "stage_local_step": event.stage_local_step,
            "sigma": {
                "actual": float(event.actual_sigma),
                "actual_next": float(event.actual_sigma_next),
                "original": float(event.original_sigma),
                "original_next": float(event.original_sigma_next),
            },
        }
        if not measured:
            self.records.append(record)
            return

        x0_video = _video_stream(x0)
        if self.measurement_mode != "x0_only":
            # The residual needs both callback tensors; a missing x0 also
            # blocks it (residual = x - x0).
            x_video = _video_stream(x)
            if x0_video is not None and x_video is not None:
                record["residual"] = self._measure_residual(event, x0_video, x_video)
            else:
                record["residual"] = _error_record("missing video tensor")
        if self.measurement_mode != "residual_only":
            if x0_video is not None:
                record["x0_signal"] = self._measure_x0(event, x0_video)
            else:
                record["x0_signal"] = _error_record("missing x0 video tensor")
        self.records.append(record)

    def on_transition(self, event) -> None:
        self.transitions.append({
            "transition_index": event.transition_index,
            "from_stage": event.from_stage,
            "to_stage": event.to_stage,
            "global_schedule_index": event.global_schedule_index,
            "from_scale": float(event.from_scale),
            "to_scale": float(event.to_scale),
            "ratio": float(event.scale_ratio),
            "sigma_before_alignment": float(event.sigma_before_alignment),
            "sigma_after_alignment": float(event.sigma_after_alignment),
            "kappa": float(event.kappa),
            "source_hw": [event.source_h, event.source_w],
            "target_hw": [event.target_h, event.target_w],
        })
        self._next_transition_index = max(self._next_transition_index, event.transition_index + 1)
        self._arm_stage(event.to_stage)

    def on_run_end(self, event) -> None:
        # Records and transitions are already in place; the document is
        # assembled at node time via build_document()/to_json().
        return None

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------
    def _arm_stage(self, stage_idx: int) -> None:
        """Set the boundary frequency + fit-prediction targets for a stage.

        The current boundary (plans §27-§28) is directly measurable by this
        stage; later boundaries are fit-extrapolation targets only (§29-§33).
        """
        self._omega_boundary = self._boundary_omega(stage_idx)
        self._future_omegas = [
            (idx, omega)
            for idx in range(stage_idx, len(self._scales) - 1)
            if (omega := self._boundary_omega(idx)) is not None
        ]

    def _boundary_omega(self, stage_idx: int) -> float | None:
        """Boundary frequency of stage `stage_idx`'s next transition (§27)."""
        if stage_idx >= len(self._scales) - 1:
            return None  # the final stage has no next transition
        return self._scales[stage_idx] * min(self._full_h, self._full_w) / 2.0

    def _measure_x0(self, event, x0_video) -> dict[str, Any]:
        """x0 signal basis: fit + direct boundary measurement + EMA (§24, §27-§36)."""
        t0 = time.perf_counter() if self.record_timing else None
        try:
            # Plan §21: fit only the canonical range, capped at half the
            # current stage's smaller spatial dimension, never DC.
            omega_max = min(event.stage_h, event.stage_w) / 2.0
            fit = fit_power_law_torch(
                *radial_dct_power_torch(x0_video),
                omega_min=DEFAULT_OMEGA_MIN,
                omega_max=omega_max,
            )
            x0_block: dict[str, Any] = {"fit": build_fit_record(fit)}
            boundary = self._boundary_block(event, x0_video)
            if boundary is not None:
                x0_block["current_boundary"] = boundary
            x0_block["fit_predictions"] = self._fit_predictions(fit)
            if self.store_radial_profiles:
                freqs, profile = radial_dct_power_torch(x0_video)
                x0_block["radial_profile"] = {
                    "freqs": [float(v) for v in freqs.detach().cpu().tolist()],
                    "power": [float(v) for v in profile.detach().cpu().tolist()],
                }
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            x0_block = _error_record(f"x0 measurement failed: {exc}")
        if t0 is not None:
            if x0_video.is_cuda:
                torch.cuda.synchronize()
            x0_block["analysis_ms"] = (time.perf_counter() - t0) * 1000.0
        return x0_block

    def _measure_residual(self, event, x0_video, x_video) -> dict[str, Any]:
        """Residual basis: fit over (x - x0) (§24 B, §37)."""
        t0 = time.perf_counter() if self.record_timing else None
        try:
            if x0_video.shape != x_video.shape:
                raise ValueError(
                    f"residual shape mismatch: x0 {list(x0_video.shape)} vs x {list(x_video.shape)}"
                )
            omega_max = min(event.stage_h, event.stage_w) / 2.0
            residual = x_video.detach() - x0_video.detach()
            freqs, profile = radial_dct_power_torch(residual)
            del residual
            fit = fit_power_law_torch(
                freqs, profile,
                omega_min=DEFAULT_OMEGA_MIN,
                omega_max=omega_max,
            )
            block: dict[str, Any] = {"fit": build_fit_record(fit)}
            # Plan §37: an independent residual-namespace boundary power +
            # log-space EMA, directly comparable with the old static
            # Harvester's residual basis. Never blended with the x0 EMA.
            boundary = self._residual_boundary_block(event, freqs, profile)
            if boundary is not None:
                block["current_boundary"] = boundary
        except (ValueError, RuntimeError, ZeroDivisionError) as exc:
            block = _error_record(f"residual measurement failed: {exc}")
        if t0 is not None:
            if x0_video.is_cuda:
                torch.cuda.synchronize()
            block["analysis_ms"] = (time.perf_counter() - t0) * 1000.0
        return block

    def _residual_boundary_block(self, event, freqs, profile) -> dict[str, Any] | None:
        """Residual-namespace boundary power + EMA (plan §37), or None when
        the stage has no next transition or the frequency is unmeasurable."""
        omega = self._omega_boundary
        if omega is None or not freqs.numel():
            return None
        direct_available = omega <= float(freqs[-1])
        power_point = sample_radial_power(freqs, profile, omega)
        power_band = sample_radial_band_power(
            freqs, profile, omega, self.boundary_band_half_width,
        )
        block: dict[str, Any] = {
            "transition_index": self._next_transition_index,
            "omega": omega,
            "direct_available": direct_available,
            "power_point": finite_or_none(power_point),
            "power_band_mean": finite_or_none(power_band),
        }
        if direct_available and power_point > 0:
            stage_ema = self._ema_residual.get(event.stage_index)
            ema_power = update_log_ema(stage_ema, power_point, self.smoothing_alpha)
            self._ema_residual[event.stage_index] = ema_power
            block["ema_power"] = finite_or_none(ema_power)
        else:
            block["ema_power"] = None
        return block

    def _boundary_block(self, event, x0_video) -> dict[str, Any] | None:
        """Direct boundary-power measurement for the stage's next transition.

        Plans §27-§31 (direct point/band power + live activation thresholds
        and eligibility) and §35 (per-stage log-space EMA of the raw x0
        boundary power, seeded from the stage's first valid measurement —
        never from the static A/beta). ``None`` when the stage has no next
        transition (final stage).
        """
        omega = self._omega_boundary
        if omega is None:
            return None
        freqs, profile = radial_dct_power_torch(x0_video)
        direct_available = bool(freqs.numel()) and omega <= float(freqs[-1])
        power_point = sample_radial_power(freqs, profile, omega)
        power_band = sample_radial_band_power(
            freqs, profile, omega, self.boundary_band_half_width,
        )
        block: dict[str, Any] = {
            "transition_index": self._next_transition_index,
            "omega": omega,
            "direct_available": direct_available,
            "power_point": finite_or_none(power_point),
            "power_band_mean": finite_or_none(power_band),
        }
        if direct_available:
            threshold_point = activation_threshold(power_point, self.delta)
            threshold_band = activation_threshold(power_band, self.delta)
            block["activation_threshold_point"] = finite_or_none(threshold_point)
            block["activation_threshold_band"] = finite_or_none(threshold_band)
            block["eligible_point"] = bool(event.actual_sigma <= threshold_point)
            block["eligible_band"] = bool(event.actual_sigma <= threshold_band)
            # Per-stage log-space EMA of the raw direct power (plan §35).
            stage_ema = self._ema_x0.get(event.stage_index)
            ema_power = update_log_ema(stage_ema, power_point, self.smoothing_alpha)
            self._ema_x0[event.stage_index] = ema_power
            ema_threshold = activation_threshold(ema_power, self.delta)
            block["ema_power"] = finite_or_none(ema_power)
            block["ema_threshold"] = finite_or_none(ema_threshold)
            block["eligible_ema"] = bool(event.actual_sigma <= ema_threshold)
        else:
            # Plan §29: never invent a direct measurement by extrapolation.
            block["ema_power"] = None
            block["ema_threshold"] = None
            block["eligible_ema"] = None
        return block

    def _fit_predictions(self, fit) -> list[dict[str, Any]]:
        """Fit-derived boundary estimates for this and future transitions.

        Plans §32-§34: the current boundary is labelled ``power_law_fit``,
        future boundaries (beyond the current stage's representable spectrum)
        are labelled ``fit_extrapolation`` (plan §29). Each entry carries the
        runtime-equivalent ``predicted_original_step`` from
        ``first_step_below(original_sigmas, threshold)``.
        """
        if fit["status"] != "ok" or not self._future_omegas:
            return []
        predictions = []
        for transition_index, omega in self._future_omegas:
            p_fit = power_at_frequency(omega, fit["A"], fit["beta"])
            threshold_fit = activation_threshold(p_fit, self.delta)
            prediction = {
                "transition_index": transition_index,
                "omega": omega,
                "P": finite_or_none(p_fit),
                "threshold": finite_or_none(threshold_fit),
                "measurement": (
                    "power_law_fit"
                    if omega == self._omega_boundary
                    else "fit_extrapolation"
                ),
            }
            if self._original_sigmas:
                prediction["predicted_original_step"] = first_step_below(
                    self._original_sigmas, threshold_fit,
                )
            predictions.append(prediction)
        return predictions

    # ------------------------------------------------------------------
    # Document assembly (plans §39, §41)
    # ------------------------------------------------------------------
    def set_original_sigmas(self, sigmas) -> None:
        """Record the user's input sigma schedule (floats).

        The runtime does not put the schedule on the events, so the node
        hands it in once before the run; `predicted_original_step` and the
        static_scheduler block read it from here.
        """
        values = [float(s) for s in sigmas]
        self._original_sigmas = values
        if self._static_scheduler:
            self._static_scheduler["original_sigmas"] = values

    def _stage_fit_series(self, stage_records, basis):
        """Per-stage scalar fit statistics for one basis (plan §41).

        Returns (beta_series, A_series, r2_series) from successfully fitted
        records only — scalar metrics, never averaged radial profiles.
        """
        betas, amps, r2s = [], [], []
        for record in stage_records:
            fit = record.get(basis, {}).get("fit", {})
            if fit.get("status") != "ok":
                continue
            betas.append(float(fit["beta"]))
            amps.append(float(fit["A"]))
            r2s.append(float(fit["r_squared"]))
        return betas, amps, r2s

    @staticmethod
    def _fit_summary_block(values):
        if not values:
            return {"first": None, "last": None, "min": None, "max": None, "mean": None}
        return {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def build_summary(self) -> dict[str, Any]:
        """Per-stage scalar summary (plan §41). No zero-padded profile averaging."""
        summary: dict[str, Any] = {}
        resolved = self._static_scheduler.get("resolved_transition_steps", [])
        scales = self._static_scheduler.get("scales", [])
        stage_indices = sorted({r["stage_index"] for r in self.records})
        for stage_idx in stage_indices:
            stage_records = [r for r in self.records if r["stage_index"] == stage_idx]
            measured = [r for r in stage_records if "x0_signal" in r or "residual" in r]
            entry: dict[str, Any] = {
                "stage": stage_idx,
                "scale": scales[stage_idx] if stage_idx < len(scales) else None,
                "measured_callbacks": len(measured),
            }
            has_next = stage_idx < len(scales) - 1
            static_step = resolved[stage_idx] if has_next and stage_idx < len(resolved) else None
            entry["static_planned_transition_step"] = static_step
            entry["first_x0_direct_eligible_step"] = _first_eligible_step(
                stage_records, ("x0_signal", "current_boundary", "eligible_point"),
            )
            entry["first_x0_ema_eligible_step"] = _first_eligible_step(
                stage_records, ("x0_signal", "current_boundary", "eligible_ema"),
            )
            # Fit-eligibility: actual sigma at or below the fit-derived
            # threshold at the current boundary frequency (plan §34).
            entry["first_x0_fit_eligible_step"] = next(
                (
                    r["global_schedule_index"]
                    for r in stage_records
                    if _fit_threshold_at_current(r) is not None
                    and r["sigma"]["actual"] <= _fit_threshold_at_current(r)
                ),
                None,
            )
            base = next(
                (
                    v
                    for v in (
                        entry["first_x0_direct_eligible_step"],
                        entry["first_x0_ema_eligible_step"],
                        entry["first_x0_fit_eligible_step"],
                    )
                    if v is not None
                ),
                None,
            )
            entry["difference_vs_static_step"] = (
                base - static_step if base is not None and static_step is not None else None
            )
            for basis, name in (("x0_signal", "x0"), ("residual", "residual")):
                betas, amps, r2s = self._stage_fit_series(measured, basis)
                entry[f"{name}_beta"] = self._fit_summary_block(betas)
                entry[f"{name}_A"] = self._fit_summary_block(amps)
                entry[f"{name}_r2_mean"] = sum(r2s) / len(r2s) if r2s else None
            summary[str(stage_idx)] = entry
        return summary

    def measurement_bases(self) -> list[str]:
        bases = []
        if self.measurement_mode != "residual_only":
            bases.append("denoised_x0_video")
        if self.measurement_mode != "x0_only":
            bases.append("residual_x_minus_x0_video")
        return bases

    def build_document(self) -> dict[str, Any]:
        """Top-level harvest document (plan §39)."""
        return {
            "schema_version": 1,
            "type": "minimax_h3_speed_sigma_harvest",
            "mode": "observational",
            "adaptive_control": False,
            "measurement_bases": self.measurement_bases(),
            "sampler": "euler",
            "static_scheduler": self._static_scheduler,
            "analysis": self._analysis_block,
            "records": self.records,
            "transitions": self.transitions,
            "summary": self.build_summary(),
        }

    def to_json(self) -> str:
        """Strict-JSON serialization (plan §42): NaN/Infinity rejected."""
        return dumps_strict(self.build_document())


def _first_eligible_step(stage_records, path):
    """First global index whose eligibility flag at `path` is True."""
    return next(
        (
            r["global_schedule_index"]
            for r in stage_records
            if _dig(r, path) is True
        ),
        None,
    )


def _video_stream(tensor: Any) -> torch.Tensor | None:
    """H3 NestedTensor unbind: the [B, C, T, H, W] video stream, or None.

    Mirrors the extraction used by the existing diagnostic nodes — the
    callback tensors may be a packed NestedTensor (video + audio) or a
    plain 5-D video tensor.
    """
    if tensor is None:
        return None
    if getattr(tensor, "is_nested", False):
        for stream in tensor.unbind():
            if isinstance(stream, torch.Tensor) and stream.ndim == 5:
                return stream
        return None
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        return tensor
    return None


def _error_record(reason: str) -> dict[str, Any]:
    """Known measurement failure (plan §43): explicit error status + reason."""
    return {"status": "error", "reason": reason}


def _dig(record, path):
    node: Any = record
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _fit_threshold_at_current(record):
    """The fit-derived threshold at the record's current boundary, if any."""
    predictions = record.get("x0_signal", {}).get("fit_predictions", [])
    for prediction in predictions:
        if prediction.get("measurement") == "power_law_fit":
            return prediction.get("threshold")
    return None


__all__ = [
    "DEFAULT_OMEGA_MIN",
    "DEFAULT_SMOOTHING_ALPHA",
    "SpeedHarvestCollector",
    "radial_dct_power_torch",
    "fit_power_law_torch",
    "sample_radial_power",
    "sample_radial_band_power",
    "update_log_ema",
    "finite_or_none",
    "build_fit_record",
    "dumps_strict",
]

