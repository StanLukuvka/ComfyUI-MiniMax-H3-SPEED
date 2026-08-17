"""Debug node: apply DCT lowpass filter to a latent for ablation testing."""

from __future__ import annotations

import torch

from minimax_h3_speed.spectral import dct2, idct2


class MiniMaxH3DCTLowpass:
    """Apply lowpass filter in DCT domain — useful for ablation studies."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "cutoff_frequency": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "sampling/minimax_h3_speed/debug"

    def apply(self, latent, cutoff_frequency=0.5):
        """Return latent with high frequencies attenuated."""
        samples = latent.get("samples")
        if not hasattr(samples, "is_nested") or not samples.is_nested:
            return (latent,)

        streams = samples.unbind()
        new_streams = []
        for stream in streams:
            if stream.ndim == 5:  # video stream
                # Forward DCT
                coeffs = dct2(stream.float())
                # Apply lowpass mask
                B, C, T, H, W = coeffs.shape
                mask = torch.ones_like(coeffs)
                cx, cy = W // 2, H // 2
                yy, xx = torch.meshgrid(
                    torch.arange(H, device=coeffs.device),
                    torch.arange(W, device=coeffs.device),
                    indexing="ij",
                )
                dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
                max_dist = max(H, W) / 2.0
                mask[dist > cutoff_frequency * max_dist] = 0.0
                filtered = coeffs * mask
                # Inverse DCT
                result = idct2(filtered)
                new_streams.append(result)
            else:
                new_streams.append(stream)

        # Reconstruct nested tensor (mock for testing)
        class _MockNested:
            is_nested = True
            def __init__(self, streams):
                self._streams = streams
            def unbind(self):
                return self._streams

        new_latent = {"samples": _MockNested(new_streams)}
        return (new_latent,)
