"""Debug node: compute and return radial power spectrum of a latent."""

from __future__ import annotations

import numpy as np
import torch

from minimax_h3_speed.spectral import dct2


class MiniMaxH3PowerSpectrum:
    """Compute radial power spectrum of a video latent for debugging."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compute"
    CATEGORY = "sampling/minimax_h3_speed/debug"
    OUTPUT_NODE = True

    def compute(self, latent):
        """Return JSON string with power spectrum data."""
        samples = latent.get("samples")
        if not hasattr(samples, "is_nested") or not samples.is_nested:
            return ("Not an H3 nested latent",)

        streams = samples.unbind()
        if len(streams) < 1:
            return ("No video stream found",)

        video = streams[0]  # [B, C, T, H, W]
        if video.ndim != 5:
            return (f"Expected 5D tensor, got {video.ndim}D",)

        # Compute DCT and radial power spectrum
        coeffs = dct2(video.float())
        power = coeffs.abs() ** 2
        power = power.mean(dim=(0, 1, 2))  # [H, W]

        H, W = power.shape
        cx, cy = W // 2, H // 2
        yy, xx = np.mgrid[0:H, 0:W]
        radial = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(int)
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

        # Return as JSON string
        import json
        report = json.dumps({
            "shape": list(video.shape),
            "freqs": freqs.tolist(),
            "power": profile.tolist(),
        })
        return (report,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PowerSpectrum": MiniMaxH3PowerSpectrum}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3PowerSpectrum": "MiniMax H3 SPEED — Power Spectrum"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3PowerSpectrum"]
