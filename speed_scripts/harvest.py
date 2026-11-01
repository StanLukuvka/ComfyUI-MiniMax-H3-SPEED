"""Measure the residual spectrum used to calibrate Automatic SPEED.

Fit ``P = A * |omega|^(-beta)`` from ``x - denoised`` on a native
full-resolution sampler run.
"""

from __future__ import annotations

import json
import logging
import math

import comfy.utils
import numpy as np
import torch

from .families import get_family, prepare_latent
from .planning import activation_threshold, find_first_step_below, power_at_frequency
from .spectral import dct2


def compute_residual(x_value, denoised_value, family) -> torch.Tensor | None:
    """Return the spatial-stream residual ``x - denoised`` when both inputs are valid."""
    x_main = family.main_stream(x_value)
    denoised_main = family.main_stream(denoised_value)
    if x_main is None or denoised_main is None:
        return None
    return x_main - denoised_main


def radial_dct_power(video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Measure average DCT power by distance from zero frequency."""
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
    """Fit A and beta to the measured frequency-power curve."""
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
    """Rate how well the power-law fit matches the measured data."""
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


log = logging.getLogger(__name__)


def _error_json(error, message, **fields):
    payload = {"error": error, "message": message, **fields}
    return json.dumps(payload)


def _sampler_label(sampler):
    """Best-effort sampler name for the report (from its function name, else its class)."""
    fn = getattr(sampler, "sampler_function", None)
    name = getattr(fn, "__name__", None) or type(sampler).__name__
    return name.removeprefix("sample_").removesuffix("_function")


def run_harvest(
    noise,
    guider,
    sampler,
    sigmas,
    latent_image: dict,
    model_type: str,
    delta: float = 0.005,
) -> tuple[str, dict]:
    """Execute native full-resolution sampling with residual spectral capture.

    Measures (x - denoised) at each step, fits P = A * |w|^-beta, and returns
    the calibration JSON string alongside the diagnostic output latent dict.

    Returns:
        tuple[str, dict]: (output_json, output_latent)
    """
    # get_family validates model_type and runs the family's own checks.
    family = get_family(model_type, guider)

    callback_count = 0
    capture_count = 0
    freqs_all = []
    profiles_all = []
    first_residual_error = None
    first_profile_error = None

    def _capture(x_current, denoised_est):
        """Turn one residual into a CPU frequency-power profile immediately."""
        nonlocal capture_count, first_residual_error, first_profile_error
        try:
            residual = compute_residual(x_current, denoised_est, family)
        except Exception as exc:
            if first_residual_error is None:
                first_residual_error = exc
                log.warning(
                    "[SPEED-harvest] residual capture failed; later repeats suppressed: %r",
                    exc,
                )
            return
        if residual is None:
            return
        capture_count += 1
        try:
            freqs, profile = radial_dct_power(residual)
        except Exception as exc:
            if first_profile_error is None:
                first_profile_error = exc
                log.warning(
                    "[SPEED-harvest] spectral profile reduction failed; later repeats suppressed: %r",
                    exc,
                )
            return
        freqs_all.append(freqs)
        profiles_all.append(profile)

    # ComfyUI calls sampler callbacks as: (step, denoised, x, total_steps).
    def _capture_callback(step, denoised, x, total_steps):
        nonlocal callback_count
        callback_count += 1
        return _capture(x, denoised)

    sampler_name = _sampler_label(sampler)
    latent_image = prepare_latent(guider, latent_image)
    latent_tensor = latent_image["samples"]
    noise_tensor = noise.generate_noise(latent_image)

    try:
        result = guider.sample(
            noise_tensor,
            latent_tensor,
            sampler,
            sigmas,
            callback=_capture_callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=getattr(noise, "seed", 42),
        )
    except Exception as exc:
        return (
            _error_json(
                "harvest_failed",
                f"Native {sampler_name} harvest failed: {exc}",
                sampler_name=sampler_name,
                fix="Check the same sampler runs without SPEED (KSampler or SamplerCustomAdvanced) with these inputs.",
            ),
            latent_image,
        )

    if capture_count == 0:
        if first_residual_error is not None:
            message = f"Residual capture failed: {first_residual_error}"
        elif callback_count:
            hint = (
                "use model_type flow_image for image models"
                if model_type == "h3_video"
                else "use model_type h3_video for MiniMax-H3 video latents"
            )
            message = (
                f"No per-step residual snapshots recorded. The sampler callback fired "
                f"{callback_count} times, but its latent does not match model_type "
                f"{model_type}; {hint}."
            )
        else:
            message = "No per-step residual snapshots recorded. The native sampler callback did not fire — check ComfyUI setup."
        return (
            _error_json(
                "no_captures",
                message,
                sampler_name=sampler_name,
                n_captures=0,
            ),
            latent_image,
        )

    if not profiles_all:
        message = (
            f"Spectral profile reduction failed: {first_profile_error}"
            if first_profile_error is not None
            else "Captured residuals produced no valid spectral profiles — residual may be zero or non-physical."
        )
        return (
            _error_json(
                "no_spectral_profiles",
                message,
                sampler_name=sampler_name,
                n_captures=capture_count,
            ),
            latent_image,
        )

    profile_mean = np.mean(np.stack(profiles_all, axis=0), axis=0)
    freqs_mean = freqs_all[0]

    try:
        fit = fit_power_law(freqs_mean, profile_mean)
    except ValueError as exc:
        return (
            _error_json(
                "fit_failed",
                f"Power-law fit failed: {exc}",
                sampler_name=sampler_name,
                n_captures=capture_count,
            ),
            latent_image,
        )

    A = float(fit["A"])
    beta = float(fit["beta"])
    r2 = float(fit["r_squared"])
    health = classify_fit_quality(fit)

    main = family.main_stream(latent_tensor)
    if main is None:
        H_full = W_full = None
    else:
        H_full, W_full = map(int, main.shape[-2:])

    # Convert sigmas once for the transition report.
    try:
        sigmas_list = [float(s) for s in sigmas]
    except Exception:
        sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

    # Automatic computes transition steps from A, beta, delta and the live sigmas.
    calibration = {
        "schema_version": 2,
        "noise_amplitude": A,
        "noise_decay_exponent": beta,
        "delta": float(delta),
        "r2": r2,
        "health": health,
        "sampler_name": sampler_name,
        # This fit measures x - denoised; x0 uses a different calibration basis.
        "measurement_basis": "residual_x_minus_denoised",
        "calibration_kind": "empirical_residual_fit",
        "model_type": model_type,
    }

    lines = [
        f"Empirical {model_type} residual calibration ({sampler_name}): noise_amplitude={A:.4f}  noise_decay_exponent={beta:.4f}  r²={r2:.4f}  health={health}",
    ]
    if health in ("suspect", "weak", "invalid"):
        lines.append(
            f"WARNING: fit is {health.upper()} — beta={beta:.4f} with "
            f"r²={r2:.4f}. Not cleanly decaying. Rerun Harvest before trusting it."
        )

    usable_fit = (
        np.isfinite(A)
        and np.isfinite(beta)
        and np.isfinite(delta)
        and A > 0.0
        and beta > 0.0
        and 0.0 < delta < 1.0
        and health != "invalid"
    )
    if usable_fit:
        lines.append(
            f"Paste into SPEED Sampler (same sampler: {sampler_name}): "
            f"Tolerance (Delta)={float(delta):.3f}, noise_amplitude={A:.4f}, "
            f"noise_decay_exponent={beta:.4f}"
        )
    else:
        lines.append(
            "Do not paste this calibration into Automatic: SPEED requires "
            "positive finite noise_amplitude and noise_decay_exponent values."
        )
    # Show where this schedule would place the 0.5x and 0.75x transitions.
    if H_full is not None and W_full is not None and sigmas_list:
        try:
            omega_max = min(H_full, W_full) / 2.0
            lines.append(f"Reference (current sigmas, {len(sigmas_list)} levels):")
            for _scale in (0.50, 0.75):
                _omega = _scale * omega_max
                _p = power_at_frequency(_omega, A, beta)
                _thr = activation_threshold(_p, float(delta))
                _step = find_first_step_below(sigmas_list, _thr)
                _sig = float(sigmas_list[_step])
                lines.append(
                    f"  {_scale:.2f}x -> sigma~{_sig:.4f} "
                    f"(step {_step})  [thr {_thr:.4f}]"
                )
        except Exception:
            pass
    report = "\n".join(lines)
    calibration["report"] = report

    output_json = json.dumps(calibration)

    # Return the sampler result as a LATENT dict.
    if result is not None:
        output_latent = latent_image.copy()
        output_latent["samples"] = result
    else:
        output_latent = latent_image
    return (output_json, output_latent)


__all__ = [
    "run_harvest",
    "compute_residual",
    "radial_dct_power",
    "fit_power_law",
    "classify_fit_quality",
]
