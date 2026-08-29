"""Sigma harvester + calibration report emitter.

Runs one native full-res Euler pass over the full sigma schedule using
`guider.sample()` (not the SPEED chain), snapshots `residual = x - denoised`
on each step, fits the radial DCT power spectrum `P = A * |omega|^(-beta)`,
and emits a flat `calibration` JSON (noise_amplitude, noise_decay_exponent,
delta, r2, health, report) to paste back into the Automatic node.
"""

from __future__ import annotations

import json

import comfy.samplers
import comfy.utils
import numpy as np
import torch

from speed_scripts.harvest import (
    radial_dct_power,
    fit_power_law,
    classify_fit_quality,
)


class MiniMaxH3HarvestToConfig:
    """Sigma harvester — native Euler pass with per-step residual capture."""

    DESCRIPTION = (
        "Sigma Harvest — run this ONCE on a full-res native Euler generation to "
        "calibrate the Automatic sampler. It measures how noise falls off with "
        "frequency (P = A·|ω|^-beta) and gives you A/beta to paste into the "
        "Automatic node. Does NOT use SPEED — it must run at full res with a "
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
        **kwargs,
    ):

        # Tolerance (Delta) is the UI label — accept delta alias for old workflows/tests
        delta = kwargs.get("Tolerance (Delta)",
                kwargs.get("Tolerance",
                kwargs.get("tolerance",
                kwargs.get("delta", kwargs.get("Delta", 0.01)))))
        delta = float(delta)

        sampler_obj = comfy.samplers.sampler_object("euler")

        residual_snapshots = []

        def _capture(sigma_val, step_idx, x_current, denoised_est):
            """Record one residual snapshot from (sigma, step, x, denoised)."""
            try:
                residual = self.compute_video_residual(x_current, denoised_est)
            except Exception:
                residual = None
            if residual is not None:
                residual_snapshots.append(
                    {
                        "step_index": int(step_idx),
                        "sigma": float(sigma_val),
                        "residual_video": residual,
                    }
                )

        # ComfyUI callback signatures across versions:
        #  - dict-arg: callback({"x", "i"/"step", "sigma", "denoised"})     (newer)
        #  - kwargs:   callback(x=..., denoised=..., i=..., sigma=...)      (mid)
        #  - legacy:   callback(step, denoised, x, total_steps)              (old)
        def _compat_callback(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], dict):
                info = args[0]
                return _capture(
                    info.get("sigma", 0.0),
                    info.get("i", info.get("step", 0)),
                    info.get("x"),
                    info.get("denoised"),
                )
            if "sigma" in kwargs and "denoised" in kwargs:
                return _capture(
                    kwargs.get("sigma", 0.0),
                    kwargs.get("i", kwargs.get("step", 0)),
                    kwargs.get("x"),
                    kwargs.get("denoised"),
                )
            # Legacy positional: (step, denoised, x, total_steps)
            if len(args) >= 3:
                step, denoised, x = args[0], args[1], args[2]
                sigma_val = float(sigmas[step]) if step < len(sigmas) else 0.0
                return _capture(sigma_val, step, x, denoised)

        # ComfyUI LATENT is always {"samples": <tensor>}; the fallback to the
        # raw input is paranoia for old test fakes that pass a tensor directly.
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
                'recorded. The native sampler callback did not fire — check ComfyUI '
                'setup.","n_captures":0}',
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
                'produced no valid spectral profiles — residual may be zero or '
                'non-physical.","n_captures":' + str(len(residual_snapshots)) + '}',
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

        # sigmas_list not needed for plug-and-play, but keep for debugging if needed
        try:
            sigmas_list = [float(s) for s in sigmas]
        except Exception:
            sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        # Plug-and-play for SPEED's delta_custom: just feed A/beta into
        # noise_amplitude / noise_decay_exponent + delta. No per-preset
        # transition_steps table — SPEED computes it via resolve_transition_steps.
        calibration = {
            "noise_amplitude": A,
            "noise_decay_exponent": beta,
            "delta": float(delta),
            "r2": r2,
            "health": health,
        }

        # Human-readable report — just the plug-and-play values
        lines = [
            f"Calibrated: noise_amplitude={A:.3f}  noise_decay_exponent={beta:.3f}  r²={r2:.4f}  health={health}",
        ]
        if health in ("suspect", "weak", "invalid"):
            lines.append(
                f"WARNING: fit is {health.upper()} — beta={beta:.3f} with "
                f"r²={r2:.4f}. Not cleanly decaying. Rerun harvest or use manual preset."
            )
        lines.append(f"Paste into SPEED Sampler: Tolerance (Delta)={float(delta):.3f}, noise_amplitude={A:.3f}, noise_decay_exponent={beta:.3f} (transition_mode=delta_custom)")
        # Diagnostic only — not part of the JSON to paste. Shows where delta_custom
        # will place the two most common reference scales for this sigmas length.
        # Derived exactly as runtime does: omega = scale * min(H,W)/2 -> P(omega) -> thr -> first step <= thr.
        try:
            from speed_scripts.h3_runtime import power_at_frequency, activation_threshold
            omega_max = min(H_full, W_full) / 2.0
            # find helper mirrors h3_runtime._find_first_step_below
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


NODE_CLASS_MAPPINGS = {"MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3HarvestToConfig": "MiniMax H3 SPEED — Sigma Harvest (Native Euler)"
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3HarvestToConfig",
]
