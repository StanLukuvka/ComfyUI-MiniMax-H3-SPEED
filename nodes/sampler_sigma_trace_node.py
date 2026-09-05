"""Sigma Trace — per-callback telemetry for one native Euler pass.

Identical inputs and H3 latent extraction as the Sigma Harvest node, but instead
of fitting a single power law across all steps, Sigma Trace records one
telemetry record per sampler callback (step index, sigma, predicted x0 shape,
spatial DCT band powers, temporal DCT summary, basic signal statistics). The
goal is to preserve the full sigma trajectory without any aggregation so the
trajectory can be plotted and analysed later for transition timing.
"""

from __future__ import annotations

import json
import math
from typing import Any

import torch

import comfy.samplers
import comfy.utils

try:
    from speed_scripts.spectral import dct2 as _dct2, dct_temporal as _dct_temporal
except Exception:
    _dct2 = None
    _dct_temporal = None

try:
    import latent_preview as _lp
except Exception:
    _lp = None


SCHEMA_VERSION = 1
MEASURED_TENSOR = "denoised_x0_video"
SPATIAL_BAND_NAMES = ("low", "mid", "high")


def _extract_video_stream(tensor: Any) -> torch.Tensor | None:
    """Return the [B, C, T, H, W] video stream from a callback's tensor or None."""
    if tensor is None:
        return None
    if hasattr(tensor, "is_nested") and tensor.is_nested:
        for stream in tensor.unbind():
            if isinstance(stream, torch.Tensor) and stream.ndim == 5:
                return stream
        return None
    if isinstance(tensor, torch.Tensor) and tensor.ndim == 5:
        return tensor
    return None


def _extract_nested_shapes(tensor: Any) -> list[list[int]] | None:
    if tensor is None or not (hasattr(tensor, "is_nested") and tensor.is_nested):
        return None
    shapes = []
    for stream in tensor.unbind():
        if isinstance(stream, torch.Tensor):
            shapes.append(list(stream.shape))
    return shapes or None


def _signal_stats(video: torch.Tensor) -> dict[str, float | int | None]:
    flat = video.detach().float()
    if flat.numel() == 0:
        return {"mean": None, "std": None, "rms": None, "min": None, "max": None, "abs_mean": None}
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    rms = float(flat.pow(2).mean().sqrt().item())
    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
    }


def _spatial_band_powers(video: torch.Tensor) -> dict[str, Any]:
    """2D DCT per (batch, channel, time) frame, then mean |coeff|^2 over three
    spatial frequency bands (low <= 1/3, mid 1/3-2/3, high > 2/3 of the smaller
    spatial axis). Returns {bands, powers[band][frame]} normalised so the three
    bands sum to 1.0.
    """
    n_frames = int(video.shape[2]) if video.ndim == 5 else 0
    empty = {
        "bands": list(SPATIAL_BAND_NAMES),
        "powers": [[0.0] for _ in SPATIAL_BAND_NAMES],
        "n_frames": 0,
    }
    if video.numel() == 0 or video.ndim != 5 or _dct2 is None:
        return empty
    H, W = int(video.shape[-2]), int(video.shape[-1])
    if H < 2 or W < 2:
        return empty
    low_cut = max(1, min(H, W) // 3)
    mid_cut = max(low_cut + 1, 2 * min(H, W) // 3)
    try:
        coeffs = _dct2(video.detach().float())
    except Exception:
        return empty
    power = coeffs.pow(2)  # real-valued after DCT
    band_slices = {
        SPATIAL_BAND_NAMES[0]: (..., slice(0, low_cut), slice(0, W)),
        SPATIAL_BAND_NAMES[1]: (..., slice(low_cut, mid_cut), slice(0, W)),
        SPATIAL_BAND_NAMES[2]: (..., slice(mid_cut, H), slice(0, W)),
    }
    powers: list[list[float]] = []
    for band_name in SPATIAL_BAND_NAMES:
        per_frame = power[band_slices[band_name]].mean(dim=(0, 1, 2, 3))  # [T]
        powers.append([float(v) for v in per_frame.detach().cpu().tolist()])
    total_energy = float(power.sum().item())
    if total_energy > 0.0:
        powers = [[p / total_energy for p in band_powers] for band_powers in powers]
    return {"bands": list(SPATIAL_BAND_NAMES), "powers": powers, "n_frames": n_frames}


def _temporal_band_summary(video: torch.Tensor) -> dict[str, Any]:
    """1D DCT over the temporal axis of the spatial-DC coefficient. Returns the
    raw temporal power spectrum (length T) so callers can plot how the temporal
    energy distribution evolves per sigma without any per-run aggregation.
    """
    if video.numel() == 0 or video.ndim != 5 or _dct_temporal is None:
        return {"available": False, "reason": "no_video_tensor", "power": []}
    T = int(video.shape[2])
    if T < 2:
        return {"available": False, "reason": "t_axis_too_short", "power": []}
    try:
        coeffs = _dct_temporal(video.detach().float())
        spatial_dc = coeffs.mean(dim=(-1, -2))
    except Exception:
        return {"available": False, "reason": "dct_failed", "power": []}
    if spatial_dc.numel() < 2:
        return {"available": False, "reason": "no_spatial_dc", "power": []}
    power = spatial_dc.pow(2).mean(dim=tuple(range(spatial_dc.ndim - 1)))
    return {
        "available": True,
        "reason": None,
        "power": [float(v) for v in power.detach().cpu().tolist()],
    }


def _coerce_sigma(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(()).item())
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sigma_list(sigmas: Any) -> list[float]:
    if isinstance(sigmas, torch.Tensor):
        return [float(v) for v in sigmas.detach().reshape(-1).cpu().tolist()]
    if isinstance(sigmas, (list, tuple)):
        out = []
        for v in sigmas:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out
    try:
        return [float(sigmas)]
    except (TypeError, ValueError):
        return []


def _n_sigma_steps(sigmas: Any) -> int:
    if isinstance(sigmas, torch.Tensor):
        return sigmas.numel()
    if isinstance(sigmas, (list, tuple)):
        return len(sigmas)
    return 0


def _build_callback(records: list, sigmas: Any) -> Any:
    """Wrap a stock `latent_preview.prepare_callback` so it still drives the
    ComfyUI progress/preview pipeline, then append one telemetry record per step.
    """
    sig = _sigma_list(sigmas)
    n_steps = max(0, len(sig) - 1)
    x0_output: dict = {}
    stock_cb = None
    if _lp is not None:
        try:
            stock_cb = _lp.prepare_callback(None, n_steps, x0_output)
        except Exception:
            stock_cb = None
    if stock_cb is None:
        pbar = comfy.utils.ProgressBar(n_steps)
        def _stock_fallback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
        stock_cb = _stock_fallback

    def _callback(step, x0, x, total_steps):
        si = int(step)
        sigma_now = sig[si] if 0 <= si < len(sig) else None
        sigma_next = sig[si + 1] if 0 <= si + 1 < len(sig) else None
        x0_nested = x0 if (hasattr(x0, "is_nested") and x0.is_nested) else None
        x0_video = _extract_video_stream(x0)
        record: dict[str, Any] = {
            "step_index": si,
            "sigma": sigma_now,
            "sigma_next": sigma_next,
            "sigma_source": "schedule" if sigma_now is not None else "missing",
            "latent_shapes": {"x0": _extract_nested_shapes(x0_nested)},
            "video_shape": list(x0_video.shape) if x0_video is not None else None,
            "signal": _signal_stats(x0_video) if x0_video is not None else {
                "mean": None, "std": None, "rms": None,
                "min": None, "max": None, "abs_mean": None,
            },
            "spatial_dct": _spatial_band_powers(x0_video) if x0_video is not None else {
                "bands": list(SPATIAL_BAND_NAMES),
                "powers": [[0.0] for _ in SPATIAL_BAND_NAMES],
                "n_frames": 0,
            },
            "temporal_dct": _temporal_band_summary(x0_video) if x0_video is not None else {
                "available": False, "reason": "no_video_tensor", "power": [],
            },
            "status": "ok",
            "measurement_note": MEASURED_TENSOR,
        }
        records.append(record)
        return stock_cb(step, x0, x, total_steps)

    return _callback, x0_output


def _open_video_stream_callback(records: list, sigmas: Any, patcher: Any) -> Any:
    """Variant for the real runtime path: pass the real patcher so ComfyUI's
    previewer is actually used and the live preview works on H3.
    """
    sig = _sigma_list(sigmas)
    n_steps = max(0, len(sig) - 1)
    x0_output: dict = {}
    stock_cb = None
    if _lp is not None and patcher is not None:
        try:
            stock_cb = _lp.prepare_callback(patcher, n_steps, x0_output)
        except Exception:
            stock_cb = None
    if stock_cb is None:
        pbar = comfy.utils.ProgressBar(n_steps)
        def _stock_fallback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)
        stock_cb = _stock_fallback

    def _callback(step, x0, x, total_steps):
        si = int(step)
        sigma_now = sig[si] if 0 <= si < len(sig) else None
        sigma_next = sig[si + 1] if 0 <= si + 1 < len(sig) else None
        x0_nested = x0 if (hasattr(x0, "is_nested") and x0.is_nested) else None
        x0_video = _extract_video_stream(x0)
        record: dict[str, Any] = {
            "step_index": si,
            "sigma": sigma_now,
            "sigma_next": sigma_next,
            "sigma_source": "schedule" if sigma_now is not None else "missing",
            "latent_shapes": {"x0": _extract_nested_shapes(x0_nested)},
            "video_shape": list(x0_video.shape) if x0_video is not None else None,
            "signal": _signal_stats(x0_video) if x0_video is not None else {
                "mean": None, "std": None, "rms": None,
                "min": None, "max": None, "abs_mean": None,
            },
            "spatial_dct": _spatial_band_powers(x0_video) if x0_video is not None else {
                "bands": list(SPATIAL_BAND_NAMES),
                "powers": [[0.0] for _ in SPATIAL_BAND_NAMES],
                "n_frames": 0,
            },
            "temporal_dct": _temporal_band_summary(x0_video) if x0_video is not None else {
                "available": False, "reason": "no_video_tensor", "power": [],
            },
            "status": "ok",
            "measurement_note": MEASURED_TENSOR,
        }
        records.append(record)
        return stock_cb(step, x0, x, total_steps)

    return _callback


def _trace_callback(patcher: Any, sigmas: torch.Tensor, records: list) -> Any:
    if patcher is None:
        return _build_callback(records, sigmas)[0]
    return _open_video_stream_callback(records, sigmas, patcher)


class MiniMaxH3SigmaTrace:
    """Sigma Trace — one telemetry record per sampler callback step."""

    DESCRIPTION = (
        "Sigma Trace — run a single native Euler pass and record one telemetry "
        "record per sampler callback step (sigma, x0 shape, spatial DCT band "
        "powers, temporal DCT summary, basic signal statistics). The resulting "
        "JSON preserves the full sigma trajectory without aggregation so you can "
        "plot when each frequency band activates later. Does NOT fit A/beta; use "
        "Sigma Harvest for that."
    )
    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("trace_json", "diagnostic_latent")
    FUNCTION = "trace"
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
        }

    def trace(self, noise, guider, sigmas, latent_image):
        sampler_obj = comfy.samplers.sampler_object("euler")
        records: list[dict[str, Any]] = []

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

        sigmas_list = _sigma_list(sigmas)

        patcher = getattr(guider, "model_patcher", None)
        callback = _trace_callback(patcher, sigmas, records)

        try:
            result = guider.sample(
                noise_tensor,
                latent_tensor,
                sampler_obj,
                sigmas,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=getattr(noise, "seed", None),
            )
        except Exception as exc:
            return (
                json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "error": "trace_failed",
                    "message": "Native Euler trace failed: " + str(exc).replace('"', "'"),
                    "expected_steps": max(0, len(sigmas_list) - 1),
                    "callback_count": len(records),
                    "records": records,
                }),
                latent_image,
            )

        expected_steps = max(0, len(sigmas_list) - 1)
        document = {
            "schema_version": SCHEMA_VERSION,
            "sampler": "euler",
            "measured_tensor": MEASURED_TENSOR,
            "sigmas": sigmas_list,
            "expected_steps": expected_steps,
            "callback_count": len(records),
            "complete": len(records) == expected_steps,
            "records": records,
        }

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
        return (json.dumps(document), output_latent)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SigmaTrace": MiniMaxH3SigmaTrace}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SigmaTrace": "MiniMax H3 SPEED — Sigma Trace (Per-Step Telemetry)",
}
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "MiniMaxH3SigmaTrace",
]
