"""Debug node: compute transition steps from power-law params."""

from __future__ import annotations

import torch

from minimax_h3_speed.h3_runtime import resolve_transition_steps
from minimax_h3_speed.config import SpeedConfig


class MiniMaxH3TransitionMath:
    """Compute transition steps from A, beta, delta for a given HxW."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "power_A": ("FLOAT", {"default": 219.48, "min": 1e-6, "max": 1e6}),
                "power_beta": ("FLOAT", {"default": 2.42, "min": 0.0, "max": 10.0}),
                "delta": ("FLOAT", {"default": 0.01, "min": 1e-6, "max": 0.999999}),
                "H": ("INT", {"default": 45, "min": 1, "max": 1024}),
                "W": ("INT", {"default": 80, "min": 1, "max": 1024}),
                "n_sigmas": ("INT", {"default": 20, "min": 5, "max": 100}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    FUNCTION = "compute"
    CATEGORY = "sampling/minimax_h3_speed/debug"
    OUTPUT_NODE = True

    def compute(
        self,
        power_A=219.48,
        power_beta=2.42,
        delta=0.01,
        H=45,
        W=80,
        n_sigmas=20,
    ):
        """Return transition step, activation time, and report string."""
        sigmas = torch.linspace(1.0, 0.025, n_sigmas + 1)
        config = SpeedConfig(
            scales=(0.5, 1.0),
            transition_steps=(5,),
            transition_mode="delta_custom",
            noise_policy="direct_coarse",
            delta=delta,
            power_A=power_A,
            power_beta=power_beta,
            transition_seed_offset=10000,
            full_latent_h=H,
            full_latent_w=W,
        )
        resolved = resolve_transition_steps(config, sigmas, H_full=H, W_full=W)
        step = resolved[0] if resolved else 0
        # Activation time is the sigma threshold
        t_star = float(sigmas[min(step, len(sigmas) - 1)])
        report = (
            f"Power-law: A={power_A:.2f}, beta={power_beta:.2f}\n"
            f"Delta={delta:.4f}, Latent={H}x{W}\n"
            f"Transition step: {step} / sigma={t_star:.4f}"
        )
        return (int(step), t_star, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3TransitionMath": MiniMaxH3TransitionMath}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3TransitionMath": "MiniMax H3 SPEED — Transition Math"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3TransitionMath"]
