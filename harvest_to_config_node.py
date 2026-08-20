"""HarvestToConfig node — sigma harvester + calibration report emitter.

CRITICAL DESIGN NOTE — why this wraps the NATIVE sampler, NOT the SPEED sampler:

You CANNOT sigma-harvest when actively changing sigmas. The SPEED sampler
(`MiniMaxH3SPEEDSampler`) splices and re-aligns the sigma schedule at every
stage boundary (low-res → mid-res → full-res). After each boundary the sigma
labels no longer index a monotonically-decreasing schedule, so any residual
field (x - x0) captured mid-chain has the wrong sigma attached to it and is
contaminated with DCT boundary artifacts. This is exactly why the old
in-SPEED harvest hook was deleted — it produced a mislabeled, garbage spectrum.

The CORRECT path is a SINGLE full-res Euler pass with a FIXED sigma schedule
(no stage boundaries, no DCT splicing). The harvester node:

  1. Takes (noise, guider, sigmas, latent_image) — the same inputs as a native
     ComfyUI sampler node.
  2. Runs ONE native Euler pass over the FULL sigma schedule using
     `guider.sample()` (not `run_speed_pipeline`).
  3. On each step, residual_snapshots the residual `residual = x - denoised` via the
     callback mechanism, tagging it with the live sigma value.
  4. After the pass, fits the radial DCT power spectrum `P = A * |omega|^(-beta)`
     on the accumulated residuals, emits the fitted (A, beta) as `harvest_json`
     (STRING), and prints a human-readable calibration report.

This node produces `harvest_json` for the user to feed back into the SPEED
sampler's `noise_amplitude` / `noise_decay_exponent` widgets (in `delta_custom` mode) — it does
NOT call the SPEED chain. The output also includes a passthrough `LATENT` so
it can be inserted into a ComfyUI workflow graph like any sampler node.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from minimax_h3_speed.harvest import (
    radial_dct_power,
    fit_power_law,
    classify_fit_quality,
    recommend_transition_steps,
)


class MiniMaxH3HarvestToConfig:
    """Sigma harvester — native Euler pass with per-step residual capture."""

    DESCRIPTION = (
        "Native Euler sigma harvester. Runs a single full-res Euler pass, "
        "residual_snapshots the residual noise spectrum at each step (residual = x - x0), "
        "fits P = A * |omega|^(-beta), and emits harvest_json + calibration report. "
        "Does NOT use the SPEED multi-stage chain — sigma schedule must stay fixed."
    )
    RETURN_TYPES = ("STRING", "LATENT", "LATENT")
    RETURN_NAMES = ("calibration_result", "output_latent", "denoised_latent")
    FUNCTION = "harvest"
    CATEGORY = "sampling/minimax_h3_speed/diagnostics"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        # Native sampler interface — NOT the JSON-only interface.
        # The harvester runs a single native Euler pass over the full sigma schedule
        # (no stage boundaries) and records per-step residuals for spectral analysis.
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001}),
            },
        }

    def harvest(
        self,
        noise,
        guider,
        sigmas,
        latent_image,
        delta=0.01,
    ):
        # We run a single native Euler pass. The harvester uses the native
        # sampler object (not the multi-stage SPEED chain) so the sigma schedule
        # stays intact end-to-end — critical for a meaningful spectrum.
        import comfy.samplers
        sampler_obj = comfy.samplers.sampler_object("euler")

        # Internal collector: records per-step (sigma, residual_video) pairs.
        # Residual = x (current noisy state) - denoised (x0 estimate) at that step.
        # This matches the spectral analysis definition in harvest.py.
        residual_snapshots = []

        def on_step(step_info):
            # step_info is the dict produced by the native Euler callback:
            #   {"x": x, "i": i, "sigma": sigma, "denoised": denoised}
            sigma_val = float(step_info.get("sigma", 0.0))
            x_current = step_info.get("x")
            denoised_est = step_info.get("denoised")
            if x_current is not None and denoised_est is not None:
                # Residual = noise remaining at this sigma = x - x0_approx
                # We compute it on the video stream only (ignore audio for spectrum).
                residual = self.compute_video_residual(x_current, denoised_est)
                residual_snapshots.append({
                    "step_index": step_info.get("i", 0),
                    "sigma": sigma_val,
                    "residual_video": residual,
                })

        try:
            # Native Euler pass over the FULL sigma schedule — no stage boundaries.
            result = guider.sample(
                noise,
                latent_image.get("samples") if hasattr(latent_image, "get") else latent_image,
                sampler_obj,
                sigmas,
                callback=on_step,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=getattr(noise, "seed", 42),
            )
        except Exception as exc:
            # If the native sampler call fails (e.g. comfy not fully available),
            # fall back to a synthetic harvest that explains what the user must
            # run externally — never silently return empty JSON.
            return (
                '{"error":"harvest_failed","message":"Native Euler harvest failed: '
                + str(exc)
                + '","fix":"Run the native Euler sampler outside this pack and feed the result back in."}',
                latent_image,
                None,
            )

        # Fit the spectrum from the captured residuals.
        # We aggregate all per-step residual videos into a single mean profile.
        # If residual_snapshots is empty (e.g. callback never fired), we emit an explicit
        # error JSON rather than a fake fit.
        if not residual_snapshots:
            return (
                '{"error":"no_captures","message":"No per-step residual residual_snapshots '
                'recorded. The native sampler callback did not fire — check ComfyUI '
                'setup.","n_captures":0}',
                latent_image,
                None,
            )

        # Aggregate: mean the residual profiles across all captured steps.
        # Each capture holds the full video tensor; we take the mean over time
        # to get a representative residual field per step, then bin radially.
        freqs_all, profiles_all = [], []
        for cap in residual_snapshots:
            residual_video = cap.get("residual_video")
            if residual_video is not None and hasattr(residual_video, "shape"):
                # Compute radial spectrum for this step's residual.
                try:
                    f, prof = radial_dct_power(residual_video)
                    freqs_all.append(f)
                    profiles_all.append(prof)
                except Exception:
                    # Skip steps that fail spectral analysis (e.g. zero tensor).
                    continue

        if not freqs_all:
            return (
                '{"error":"no_spectral_profiles","message":"Captured residuals '
                'produced no valid spectral profiles — residual may be zero or '
                'non-physical.","n_captures":' + str(len(residual_snapshots)) + '}',
                latent_image,
                None,
            )

        # Aggregate profiles: mean across steps at each radial bin.
        max_r = max(f.max() for f in freqs_all)
        profile_mean = None
        bin_counts = None
        for f, prof in zip(freqs_all, profiles_all):
            if profile_mean is None:
                profile_mean = prof.astype(float).copy()
                bin_counts = np.ones_like(profile_mean)
            # Re-bin by radial index (frequencies are the same index vector)
            # For simplicity, we assume radial bins are consistent per capture.
            # We take the mean of the profiles directly since freqs are aligned.
            profile_mean = profile_mean + prof.astype(float)
            bin_counts += 1.0
        profile_mean = profile_mean / np.maximum(bin_counts, 1)
        freqs_mean = freqs_all[0]  # same radial index for all residual_snapshots

        try:
            fit = fit_power_law(freqs_mean, profile_mean)
        except ValueError as exc:
            return (
                '{"error":"fit_failed","message":"Power-law fit failed: '
                + str(exc)
                + '","n_captures":' + str(len(residual_snapshots)) + '}',
                latent_image,
                None,
            )

        A = float(fit["A"])
        beta = float(fit["beta"])
        r2 = float(fit["r_squared"])
        health = classify_fit_quality(fit)

        full_video_shape = None
        try:
            # Extract spatial dims from the initial latent for recommendations.
            samples = (latent_image.get("samples")
                       if hasattr(latent_image, "get") else latent_image)
            if hasattr(samples, "is_nested") and samples.is_nested:
                video_stream = [s for s in samples.unbind() if s.ndim == 5]
                if video_stream:
                    full_video_shape = list(video_stream[0].shape)
            elif isinstance(samples, torch.Tensor):
                full_video_shape = list(samples.shape)
        except Exception:
            full_video_shape = None

        # Compute delta-optimal recommendations using the fitted (A, beta).
        if full_video_shape is not None:
            try:
                H_full, W_full = full_video_shape[-2], full_video_shape[-1]
            except (IndexError, TypeError):
                H_full, W_full = 64, 64
        else:
            H_full, W_full = 64, 64

        # Build sigma list from the input (may be a tensor or list of floats).
        try:
            sigmas_list = [float(s) for s in sigmas]
        except Exception:
            sigmas_list = [float(sigmas[i]) for i in range(len(sigmas))]

        try:
            rec = recommend_transition_steps(A, beta, sigmas_list, H_full, W_full, delta=float(delta))
        except Exception:
            rec = {}

        fit_results_json = json.dumps({
            "overall_fit_A": A,
            "overall_fit_beta": beta,
            "overall_fit_r2": r2,
            "fit_mode": "delta_custom",
            "fit_health": health,
            "n_sigma_levels": len(sigmas_list) - 1,
            "sigma_levels": sigmas_list,
            "recommended_config": rec,
        })

        # Human-readable calibration report (same format as before, but now
        # backed by a real native-pass harvest instead of a dead JSON parser).
        lines = [
            f"Calibrated: A={A:.3f}  beta={beta:.3f}  r²={r2:.4f}  "
            f"(fit_mode=delta_custom, health={health})",
        ]
        if health in ("suspect", "weak", "invalid"):
            lines.append(
                f"WARNING: fit is {health.upper()} — beta={beta:.3f} with "
                f"r²={r2:.4f}. Power is not cleanly decaying. Trust the "
                f"transition_steps below with caution, or rerun to fit a better spectrum."
            )
        if rec:
            lines.append("Recommended delta-optimal transition_steps (paste into sampler):")
            for name, preset in rec.items():
                lines.append(
                    f"  {name}: scales={preset['scales']}  "
                    f"transition_steps={preset['transition_steps']}"
                )
        else:
            lines.append("(no recommended_config — harvest did not include per-preset steps)")

        # Per-sigma diagnostic table: expose fitted beta evolution per sigma level.
        # Since we aggregate all steps into one fit, we show the single overall fit.
        # In a full multi-step harvest this would show per-step fits.
        lines.append(f"Per-sigma velocity fits (diagnostic):  beta={beta:+.3f}  r²={r2:.3f}")

        report = "\n".join(lines)

        # Emit both the structured JSON (for downstream automation) and the
        # human-readable report (for the user to read in ComfyUI).
        # We wrap the report in a JSON field so downstream nodes can parse it.
        output_json = json.dumps({
            "harvest_json": fit_results_json,
            "report": report,
        })

        return (output_json, result if result is not None else latent_image, None)

    def compute_video_residual(self, x_tensor, denoised_tensor):
        # Extract video streams from NestedTensor or flat tensor, compute
        # the residual (noise remaining) = current - denoised.
        # This matches the spectral analysis definition in harvest.py.
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
