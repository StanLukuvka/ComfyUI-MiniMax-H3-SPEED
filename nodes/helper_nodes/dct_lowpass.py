"""Debug node: apply DCT lowpass filter to a latent for ablation testing."""

from __future__ import annotations

from minimax_h3_speed.spectral import lowpass_filter_dct
from nodes.common import MockNested


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
        """Return latent with high frequencies attenuated (shape-preserving)."""
        samples = latent.get("samples")
        if not hasattr(samples, "is_nested") or not samples.is_nested:
            return (latent,)

        streams = samples.unbind()
        new_streams = [
            lowpass_filter_dct(stream.float(), cutoff_frequency)
            if stream.ndim == 5 else stream
            for stream in streams
        ]

        return ({"samples": MockNested(new_streams)},)
