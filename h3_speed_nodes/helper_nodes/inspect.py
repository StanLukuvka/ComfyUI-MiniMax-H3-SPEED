"""Debug node: inspect latent geometry, device, dtype."""

from __future__ import annotations

import comfy


class MiniMaxH3Inspect:
    """Debug node — prints latent shape, device, dtype for troubleshooting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    CATEGORY = "sampling/minimax_h3_speed/debug"
    OUTPUT_NODE = True

    def inspect(self, latent):
        """Return diagnostic string about the latent tensor."""
        samples = latent.get("samples")
        if hasattr(samples, "is_nested") and samples.is_nested:
            streams = samples.unbind()
            parts = []
            for i, s in enumerate(streams):
                parts.append(
                    f"stream[{i}] shape={tuple(s.shape)} "
                    f"device={s.device} dtype={s.dtype}"
                )
            report = "\n".join(parts)
        else:
            report = (
                f"shape={tuple(samples.shape)} "
                f"device={samples.device} dtype={samples.dtype}"
            )
        return (report,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3Inspect": MiniMaxH3Inspect}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3Inspect": "MiniMax H3 SPEED — Inspect"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3Inspect"]
