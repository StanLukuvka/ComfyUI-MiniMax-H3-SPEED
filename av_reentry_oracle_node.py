"""Debug node: compute audio-video reentry schedule."""

from __future__ import annotations

import torch


class MiniMaxH3AVReentryOracle:
    """Compute when audio should re-enter based on sigma schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "audio_shift": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("SIGMAS", "STRING")
    FUNCTION = "compute"
    CATEGORY = "sampling/minimax_h3_speed/debug"

    def compute(self, sigmas, audio_shift=0.1):
        """Return adjusted sigmas and report string."""
        sigmas_list = list(sigmas)
        # Find first sigma where audio can re-enter
        reentry_idx = None
        for i, s in enumerate(sigmas_list):
            if s <= audio_shift:
                reentry_idx = i
                break

        if reentry_idx is None:
            report = f"No reentry point found (audio_shift={audio_shift} not reached)"
            return (sigmas, report)

        report = (
            f"Audio re-enters at step {reentry_idx} "
            f"(sigma={sigmas_list[reentry_idx]:.4f})"
        )
        return (sigmas, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3AVReentryOracle": MiniMaxH3AVReentryOracle}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3AVReentryOracle": "MiniMax H3 SPEED — AV Reentry Oracle"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3AVReentryOracle"]
