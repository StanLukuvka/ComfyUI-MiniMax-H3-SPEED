"""Sigma harvester test variant — class renamed with `_test` suffix.

Same body as `sampler_sigma_manual_class.py`; the `_test` suffix keeps the test
import out of the live NODE_CLASS_MAPPINGS. The harvester itself does not
exercise the SPEED chain, so this mirrors the original exactly.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from speed_scripts.harvest import (
    classify_fit_quality,
    fit_power_law,
    radial_dct_power,
)


class MiniMaxH3HarvestToConfigClassTest:
    """Test variant of the LatentClass-edition sigma harvester.

    Behaviour identical to `MiniMaxH3HarvestToConfigClass`. The `_test` suffix
    keeps it out of the production NODE_CLASS_MAPPINGS export.
    """

    DESCRIPTION = (
        "TEST variant: Sigma Harvest — run this ONCE on a full-res native Euler "
        "generation to calibrate the Automatic sampler. Exposed for tests only."
    )
    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("calibration", "diagnostic_latent")
    FUNCTION = "harvest"
    CATEGORY = "sampling/minimax_h3_speed/_test"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "Tolerance (Delta)": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
            },
        }

    def harvest(self, noise, guider, sigmas, latent_image, **kwargs):

        import comfy.samplers

        delta = kwargs.get("Tolerance (Delta)",
                kwargs.get("Tolerance",
                kwargs.get("tolerance",
                kwargs.get("delta", kwargs.get("Delta", 0.01)))))
        delta = float(delta)

        sampler_obj = comfy.samplers.sampler_object("euler")

        residual_snapshots = []

        def on_step(step, denoised, x, total_steps):
            try:
                sigma_val = float(sigmas[step]) if step < len(sigmas) else 0.0
            except Exception:
                sigma_val = 0.0
            try:
                residual = self.compute_video_residual(x, denoised)
            except Exception:
                residual = None
            if residual is not None:
                residual_snapshots.append(
                    {
                        "step_index": int(step),
                        "sigma": sigma_val,
                        "residual_video": residual,
                    }
                )

        def _compat_callback(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict):
                info = args[0]
                sigma_val = float(info.get("sigma", 0.0))
                step_idx = int(info.get("i", info.get("step", 0)))
                x_current = info.get("x")
                denoised_est = info.get("denoised")
                if x_current is not None and denoised_est is not None:
                    residual = self.compute_video_residual(x_current, denoised_est)
                    if residual is not None:
                        residual_snapshots.append(
                            {
                                "step_index": step_idx,
                                "sigma": sigma_val,
                                "residual_video": residual,
                            }
                        )
                return
            if kwargs and "sigma" in kwargs:
                sigma_val = float(kwargs.get("sigma", 0.0))
                step_idx = int(kwargs.get("i", kwargs.get("step", 0)))
                x_current = kwargs.get("x")
                denoised_est = kwargs.get("denoised")
                if x_current is not None and denoised_est is not None:
                    residual = self.compute_video_residual(x_current, denoised_est)
                    if residual is not None:
                        residual_snapshots.append(
                            {
                                "step_index": step_idx,
                                "sigma": sigma_val,
                                "residual_video": residual,
                            }
                        )
                return
            return on_step(*args, **kwargs)

        if isinstance(latent_image, dict) and "samples" in latent_image:
            latent_tensor = latent_image["samples"]
        elif hasattr(latent_image, "get"):
            try:
                latent_tensor = latent_image.get("samples", latent_image)
                if latent_tensor is None:
                    latent_tensor = latent_image
            except Exception:
                latent_tensor = latent_image
        else:
            latent_tensor = latent_image

        if hasattr(noise, "generate_noise"):
            try:
                noise_tensor = noise.generate_noise(latent_image)
            except Exception:
                try:
                    noise_tensor = noise.generate_noise({"samples": latent_tensor})
                except Exception:
                    noise_tensor = noise
        else:
            noise_tensor = noise

        try:
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
                '{"error":"harvest_failed","message":"Native Euler harvest failed: '
                + str(exc).replace('"', "'")
                + '","fix":"Run the native Euler sampler outside this pack and feed the result back in."}',
                latent_image,
            )

        if not residual_snapshots:
            return (
                '{"error":"no_captures","message":"No per-step residual snapshots '
                'recorded. The native sampler callback did not fire.","n_captures":0}',
                latent_image,
            )

        freqs_all, profiles_all = [], []
        for cap in residual_snapshots:
            residual_video = cap.get("residual_video")
            if residual_video is not None and hasattr(residual_video, "shape"):
                try:
                    f, prof = radial_dct_power(residual_video)
                    freqs_all.append(f)
                    profiles_all.append(prof)
                except Exception:
                    continue

        if not freqs_all:
            return (
                '{"error":"no_spectral_profiles","message":"Captured residuals '
                'produced no valid spectral profiles.","n_captures":' + str(len(residual_snapshots)) + '}',
                latent_image,
            )

        max_len = max(len(p) for p in profiles_all)
        padded = []
        for prof in profiles_all:
            if len(prof) < max_len:
                pad = np.zeros(max_len, dtype=float)
                pad[: len(prof)] = prof.astype(float)
                padded.append(pad)
            else:
                padded.append(prof.astype(float))
        profile_mean = np.mean(np.stack(padded, axis=0), axis=0)
        freqs_mean = freqs_all[0]
        if len(freqs_mean) < max_len:
            freqs_mean = np.arange(max_len)

        try:
            fit = fit_power_law(freqs_mean, profile_mean)
        except ValueError as exc:
            return (
                '{"error":"fit_failed","message":"Power-law fit failed: '
                + str(exc).replace('"', "'")
                + '","n_captures":' + str(len(residual_snapshots)) + '}',
                latent_image,
            )

        A = float(fit["A"])
        beta = float(fit["beta"])
        r2 = float(fit["r_squared"])
        health = classify_fit_quality(fit)

        full_video_shape = None
        try:
            samples = latent_tensor
            if hasattr(samples, "is_nested") and samples.is_nested:
                video_stream = [s for s in samples.unbind() if s.ndim == 5]
                if video_stream:
                    full_video_shape = list(video_stream[0].shape)
            elif isinstance(samples, torch.Tensor):
                full_video_shape = list(samples.shape)
        except Exception:
            full_video_shape = None

        if full_video_shape is not None:
            try:
                H_full, W_full = full_video_shape[-2], full_video_shape[-1]
            except (IndexError, TypeError):
                H_full, W_full = 64, 64
        else:
            H_full, W_full = 64, 64

        try:
            sigmas_list = [float(s) for s in sigmas]
        except Exception:
            sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        calibration = {
            "noise_amplitude": A,
            "noise_decay_exponent": beta,
            "delta": float(delta),
            "r2": r2,
            "health": health,
        }

        lines = [
            f"Calibrated: noise_amplitude={A:.3f}  noise_decay_exponent={beta:.3f}  r2={r2:.4f}  health={health}",
        ]
        if health in ("suspect", "weak", "invalid"):
            lines.append(
                f"WARNING: fit is {health.upper()} — beta={beta:.3f} with "
                f"r2={r2:.4f}. Not cleanly decaying. Rerun harvest or use manual preset."
            )
        lines.append(f"Paste into SPEED Sampler: Tolerance (Delta)={float(delta):.3f}, noise_amplitude={A:.3f}, noise_decay_exponent={beta:.3f}")
        try:
            from speed_scripts.h3_runtime import power_at_frequency, activation_threshold
            omega_max = min(H_full, W_full) / 2.0
            def _first_step_below(thr: float) -> tuple[int, float]:
                for idx, s in enumerate(sigmas_list[:-1]):
                    if float(s) <= thr:
                        return idx, float(s)
                return len(sigmas_list) - 1, float(sigmas_list[-1]) if sigmas_list else 0.0
            lines.append(f"Reference (current sigmas, {len(sigmas_list)} levels):")
            for _scale in (0.50, 0.75):
                _omega = _scale * omega_max
                _p = power_at_frequency(_omega, A, beta)
                _thr = activation_threshold(_p, float(delta))
                _step, _sig = _first_step_below(_thr)
                lines.append(f"  {_scale:.2f}x -> sigma~{_sig:.4f} (step {_step})  [thr {_thr:.4f}]")
        except Exception:
            pass
        report = "\n".join(lines)
        calibration["report"] = report

        output_json = json.dumps(calibration)

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
        import torch

        def extract_video_stream(t):
            if hasattr(t, "is_nested") and t.is_nested:
                vids = [s for s in t.unbind() if s.ndim == 5]
                return vids[0] if vids else None
            if isinstance(t, torch.Tensor) and t.ndim == 5:
                return t
            return None

        x_vid = extract_video_stream(x_tensor)
        d_vid = extract_video_stream(denoised_tensor)
        if x_vid is not None and d_vid is not None:
            return x_vid - d_vid
        if isinstance(x_tensor, torch.Tensor) and isinstance(denoised_tensor, torch.Tensor):
            if x_tensor.shape == denoised_tensor.shape and x_tensor.ndim == 5:
                return x_tensor - denoised_tensor
        return None


NODE_CLASS_MAPPINGS = {"MiniMaxH3HarvestToConfigClassTest": MiniMaxH3HarvestToConfigClassTest}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3HarvestToConfigClassTest": "MiniMax H3 SPEED — Sigma Harvest (Native Euler, LatentClass) [TEST]"
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3HarvestToConfigClassTest",
]
