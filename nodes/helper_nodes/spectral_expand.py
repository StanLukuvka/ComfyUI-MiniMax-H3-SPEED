"""Debug node: visualize spectral expansion effect on noise."""

from __future__ import annotations

import torch

from minimax_h3_speed.spectral import spectral_expand_dct


class MiniMaxH3SpectralExpand:
    """Show spectral expansion effect on a noise tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "direction": (["up", "down"],),
            }
        }

    RETURN_TYPES = ("NOISE", "STRING")
    FUNCTION = "expand"
    CATEGORY = "sampling/minimax_h3_speed/debug"
    OUTPUT_NODE = True

    def expand(self, noise, sigma, direction):
        """Return expanded noise and a description string."""
        # noise is a dict with 'samples' key containing nested tensor
        samples = noise.get("samples")
        if not hasattr(samples, "is_nested") or not samples.is_nested:
            return (noise, "Not an H3 nested latent")

        streams = samples.unbind()
        if len(streams) < 1:
            return (noise, "No video stream")

        video = streams[0]  # [B, C, T, H, W]
        # Expand to double resolution
        new_H, new_W = video.shape[-2] * 2, video.shape[-1] * 2
        try:
            expanded = spectral_expand_dct(video, (new_H, new_W), sigma, seed=1000)
        except Exception as e:
            return (noise, f"Expansion failed: {e}")

        class _MockNested:
            is_nested = True
            def __init__(self, s):
                self._streams = s
            def unbind(self):
                return self._streams

        new_samples = _MockNested([expanded] + list(streams[1:]))
        new_noise = {"samples": new_samples}
        report = (
            f"Expanded {direction} from {video.shape[-2]}x{video.shape[-1]} "
            f"to {new_H}x{new_W} at sigma={sigma}"
        )
        return (new_noise, report)
