"""Calibrate Automatic SPEED from one native full-resolution sampler run.

Measure `x - denoised` at each step, fit `P = A * |omega|^(-beta)`, and
return A, beta, and delta for the Automatic node. This measures H3's residual
spectrum, not the clean-data spectrum from the SPEED paper.
"""

from __future__ import annotations

import json
import logging

import comfy.samplers
import comfy.utils
import numpy as np

from speed_scripts.harvest import (
    classify_fit_quality,
    compute_video_residual,
    extract_video_stream,
    fit_power_law,
    radial_dct_power,
)
from speed_scripts.planning import (
    activation_threshold,
    find_first_step_below,
    power_at_frequency,
)
from speed_scripts.sampler_support import SUPPORTED_SPEED_SAMPLERS


log = logging.getLogger(__name__)


def _error_json(error, message, **fields):
    payload = {"error": error, "message": message, **fields}
    return json.dumps(payload)


class MiniMaxH3HarvestToConfig:
    """Measure one native sampler run and return an Automatic calibration."""

    DESCRIPTION = (
        "Sigma Harvest calibrates Automatic from one native full-resolution "
        "sampler run. Re-run it when the sampler, model, scheduler, step count, "
        "or denoising behavior changes. It fits the residual spectrum "
        "(x - denoised) to P = A·|ω|^-beta and returns A/beta for Automatic. "
        "The calibration basis is the H3 residual spectrum, separate from the "
        "clean-data spectrum in the SPEED paper."
    )
    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("calibration", "diagnostic_latent")
    FUNCTION = "harvest"
    CATEGORY = "sampling/minimax_h3_speed/diagnostics"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "sampler_name": (list(SUPPORTED_SPEED_SAMPLERS), {"default": "euler"}),
            },
            "optional": {
                "Tolerance (Delta)": ("FLOAT", {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001}),
            },
        }

    def harvest(
        self,
        noise,
        guider,
        sigmas,
        latent_image,
        sampler_name,
        **kwargs,
    ):

        delta = float(kwargs.pop("Tolerance (Delta)", 0.005))
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Harvest option(s): {unexpected}")

        capture_count = 0
        freqs_all = []
        profiles_all = []
        first_residual_error = None
        first_profile_error = None

        def _capture(x_current, denoised_est):
            """Turn one residual into a CPU frequency-power profile immediately."""
            nonlocal capture_count, first_residual_error, first_profile_error
            try:
                residual = compute_video_residual(x_current, denoised_est)
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
            return _capture(x, denoised)

        latent_tensor = latent_image["samples"]
        noise_tensor = noise.generate_noise(latent_image)

        try:
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
            result = guider.sample(
                noise_tensor,
                latent_tensor,
                sampler_obj,
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
                    fix=f"Run the native {sampler_name} sampler outside this pack and feed the result back in.",
                ),
                latent_image,
            )

        if capture_count == 0:
            message = (
                f"Residual capture failed: {first_residual_error}"
                if first_residual_error is not None
                else "No per-step residual snapshots recorded. The native sampler callback did not fire — check ComfyUI setup."
            )
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

        video_stream = extract_video_stream(latent_tensor)
        if video_stream is None:
            H_full = W_full = None
        else:
            H_full, W_full = map(int, video_stream.shape[-2:])

        # Convert sigmas once for the transition report.
        try:
            sigmas_list = [float(s) for s in sigmas]
        except Exception:
            sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        # Automatic uses A, beta, and delta to calculate transition steps
        # from the live sigma schedule.
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
            "calibration_kind": "empirical_h3_residual_fit",
        }

        lines = [
            f"Empirical H3 residual calibration ({sampler_name}): noise_amplitude={A:.4f}  noise_decay_exponent={beta:.4f}  r²={r2:.4f}  health={health}",
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
                f"Paste into SPEED Sampler: sampler_name={sampler_name}, "
                f"Tolerance (Delta)={float(delta):.3f}, noise_amplitude={A:.4f}, "
                f"noise_decay_exponent={beta:.4f} (transition_mode=delta_custom)"
            )
        else:
            lines.append(
                "Do not paste this calibration into Automatic: SPEED requires "
                "positive finite noise_amplitude and noise_decay_exponent values."
            )
        # For reference, show where this sigma schedule would place the
        # 0.5x and 0.75x transitions.
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

        # ComfyUI expects a LATENT dict, so put the sampler result back under
        # "samples" before returning it.
        if result is not None:
            output_latent = latent_image.copy()
            output_latent["samples"] = result
        else:
            output_latent = latent_image
        return (output_json, output_latent)

NODE_CLASS_MAPPINGS = {"MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Sigma Harvest (Native Sampler)"
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3HarvestToConfig",
]
