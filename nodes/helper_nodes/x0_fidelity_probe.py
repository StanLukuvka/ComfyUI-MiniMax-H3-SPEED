"""Debug node: probe X0 fidelity during sampling."""

from __future__ import annotations

import torch


class MiniMaxH3XFidelityProbe:
    """Measure X0 fidelity (correlation between predicted and true denoised)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x0": ("LATENT",),
                "x_noisy": ("LATENT",),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    FUNCTION = "probe"
    CATEGORY = "sampling/minimax_h3_speed/debug"
    OUTPUT_NODE = True

    def probe(self, x0, x_noisy, sigma):
        """Return fidelity score and report string."""
        s0 = x0.get("samples")
        s1 = x_noisy.get("samples")
        if not hasattr(s0, "is_nested") or not s0.is_nested:
            return (0.0, "Not an H3 nested latent")

        v0 = s0.unbind()[0]
        v1 = s1.unbind()[0]
        # Fidelity = correlation between denoised predictions
        # Simulated as cosine similarity
        v0_flat = v0.flatten(1)
        v1_flat = v1.flatten(1)
        dot = (v0_flat * v1_flat).sum(dim=1)
        norm0 = v0_flat.norm(dim=1)
        norm1 = v1_flat.norm(dim=1)
        fidelity = (dot / (norm0 * norm1 + 1e-8)).mean().item()
        report = f"X0 fidelity at sigma={sigma}: {fidelity:.4f}"
        return (fidelity, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3XFidelityProbe": MiniMaxH3XFidelityProbe}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3XFidelityProbe": "MiniMax H3 SPEED — X0 Fidelity Probe"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3XFidelityProbe"]
