"""Sigma harvester + calibration report emitter.

Native full-res sampler pass over the full sigma schedule using
`guider.sample()` (not the SPEED chain), snapshots `residual = x - denoised`
on each step, immediately reduces it to a radial DCT power profile, then fits
`P = A * |omega|^(-beta)`,
and emits a flat `calibration` JSON (schema_version, sampler_name,
noise_amplitude,
noise_decay_exponent, delta, r2, health, measurement_basis,
calibration_kind, report) to paste back into the Automatic node.

The fit is an empirical H3 residual calibration (basis:
`residual_x_minus_denoised`); its A/beta are not the clean-data power
spectrum from the SPEED paper.
"""

from __future__ import annotations

import json

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


def _error_json(error, message, **fields):
    payload = {"error": error, "message": message, **fields}
    return json.dumps(payload)


class MiniMaxH3HarvestToConfig:
    """Sigma harvester — native sampler pass with per-step residual capture."""

    DESCRIPTION = (
        "Sigma Harvest — run this on a full-res native sampler generation to "
        "calibrate the Automatic sampler. Re-run it when the sampler, model, "
        "scheduler, step count, or other denoising behavior changes. It is an empirical H3 residual "
        "calibration: it measures how the residual (x - denoised) falls off with "
        "frequency (P = A·|ω|^-beta) and gives you A/beta to paste into the "
        "Automatic node. It does NOT measure the clean-data power spectrum from "
        "the SPEED paper. Does NOT use SPEED — it must run at full res with a "
        "fixed sigma schedule."
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
                "Tolerance (Delta)": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
            },
        }

    def harvest(
        self,
        noise,
        guider,
        sigmas,
        latent_image,
        sampler_name="euler",
        **kwargs,
    ):

        # Tolerance (Delta) is the UI label — accept delta alias for old workflows/tests
        delta = kwargs.get("Tolerance (Delta)",
                kwargs.get("Tolerance",
                kwargs.get("tolerance",
                kwargs.get("delta", kwargs.get("Delta", 0.01)))))
        delta = float(delta)

        capture_count = 0
        freqs_all = []
        profiles_all = []

        def _capture(x_current, denoised_est):
            """Reduce one residual to its CPU spectral profile immediately."""
            nonlocal capture_count
            try:
                residual = compute_video_residual(x_current, denoised_est)
            except Exception:
                return
            if residual is None:
                return
            capture_count += 1
            try:
                freqs, profile = radial_dct_power(residual)
            except Exception:
                return
            freqs_all.append(freqs)
            profiles_all.append(profile)

        # ComfyUI callback signatures across versions:
        #  - dict-arg: callback({"x", "i"/"step", "sigma", "denoised"})     (newer)
        #  - kwargs:   callback(x=..., denoised=..., i=..., sigma=...)      (mid)
        #  - legacy:   callback(step, denoised, x, total_steps)              (old)
        def _compat_callback(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict):
                info = args[0]
                return _capture(info.get("x"), info.get("denoised"))
            if "sigma" in kwargs and "denoised" in kwargs:
                return _capture(kwargs.get("x"), kwargs.get("denoised"))
            # Legacy positional: (step, denoised, x, total_steps)
            if len(args) >= 3:
                _step, denoised, x = args[0], args[1], args[2]
                return _capture(x, denoised)

        # ComfyUI LATENT is normally {"samples": <tensor>}; keep the raw-input
        # fallback for older tests and compatibility callers.
        latent_tensor = (
            latent_image["samples"]
            if isinstance(latent_image, dict) and "samples" in latent_image
            else latent_image
        )
        try:
            noise_tensor = noise.generate_noise(latent_image)
        except Exception:
            try:
                noise_tensor = noise.generate_noise({"samples": latent_tensor})
            except Exception:
                noise_tensor = noise

        try:
            sampler_obj = comfy.samplers.sampler_object(sampler_name)
            result = guider.sample(
                noise_tensor,
                latent_tensor,
                sampler_obj,
                sigmas,
                callback=_compat_callback,
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
            return (
                _error_json(
                    "no_captures",
                    "No per-step residual snapshots recorded. The native sampler callback did not fire — check ComfyUI setup.",
                    sampler_name=sampler_name,
                    n_captures=0,
                ),
                latent_image,
            )

        if not profiles_all:
            return (
                _error_json(
                    "no_spectral_profiles",
                    "Captured residuals produced no valid spectral profiles — residual may be zero or non-physical.",
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

        # sigmas_list not needed for plug-and-play, but keep for debugging if needed
        try:
            sigmas_list = [float(s) for s in sigmas]
        except Exception:
            sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        # Plug-and-play for SPEED's delta_custom: just feed A/beta into
        # noise_amplitude / noise_decay_exponent + delta. No per-preset
        # precomputed transition table — SPEED computes it via resolve_transition_steps.
        calibration = {
            "schema_version": 2,
            "noise_amplitude": A,
            "noise_decay_exponent": beta,
            "delta": float(delta),
            "r2": r2,
            "health": health,
            "sampler_name": sampler_name,
            # Measurement basis: this fit comes from the residual (x - denoised),
            # not the clean-data x0 spectrum. Kept alongside the original keys
            # so existing consumers keep working unchanged.
            "measurement_basis": "residual_x_minus_denoised",
            "calibration_kind": "empirical_h3_residual_fit",
        }

        # Human-readable report — just the plug-and-play values
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
        # Diagnostic only — not part of the JSON to paste. Shows where delta_custom
        # will place the two most common reference scales for this sigmas length.
        # Derived exactly as runtime does: omega = scale * min(H,W)/2 -> P(omega) -> thr -> first step <= thr.
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

        # guider.sample returns a NestedTensor/tensor, but ComfyUI LATENT is a dict
        # {"samples": ...}. Wrap it so downstream VAE decode works (otherwise
        # VAEDecodeAudio does NestedTensor["samples"] -> IndexError).
        if result is not None:
            if isinstance(latent_image, dict):
                output_latent = latent_image.copy()
                output_latent["samples"] = result
            elif isinstance(result, dict) and "samples" in result:
                output_latent = result
            else:
                output_latent = {"samples": result}
        else:
            output_latent = latent_image
        return (output_json, output_latent)

    def compute_video_residual(self, x_tensor, denoised_tensor):
        """Compatibility wrapper around the shared harvest helper."""
        return compute_video_residual(x_tensor, denoised_tensor)

NODE_CLASS_MAPPINGS = {"MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Sigma Harvest (Native Sampler)"
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3HarvestToConfig",
]
