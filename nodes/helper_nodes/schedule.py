"""Schedule node — computes SpeedConfig from sigmas + preset + mode."""

from __future__ import annotations

import torch

from minimax_h3_speed.config import SCALE_PRESETS, preset_config, SpeedConfig
from minimax_h3_speed.flow import aligned_speed_sigma
from minimax_h3_speed.h3_runtime import resolve_transition_steps


class MiniMaxH3SPEEDSchedule:
    DESCRIPTION = (
        "Computes a SpeedConfig (scales + delta-optimal transition_steps) from "
        "the sigma schedule, preset, and calibration (A, β). Use 'manual_sigma' "
        "to force a boundary sigma, or 'delta_custom' with harvested A/β for "
        "automatic tuning."
    )
    RETURN_TYPES = ("H3_SPEED_CONFIG", "STRING")
    RETURN_NAMES = ("config", "report")
    FUNCTION = "plan"
    CATEGORY = "sampling/minimax_h3_speed/schedule"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "explicit_preset": (list(SCALE_PRESETS.keys()),),
                "transition_mode": (["manual_step", "manual_sigma", "delta_custom"],),
                "noise_policy": (["direct_coarse", "coupled_full_grid"],),
                "manual_sigma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
                "delta": ("FLOAT", {"default": 0.01, "min": 0.000001, "max": 0.999999, "step": 0.001}),
                "power_A": ("FLOAT", {"default": 150.0, "min": 0.000001, "max": 1000000.0}),
                "power_beta": ("FLOAT", {"default": 2.0, "min": 0.000001, "max": 10.0}),
                "full_latent_h": ("INT", {"default": 45, "min": 1, "max": 4096}),
                "full_latent_w": ("INT", {"default": 80, "min": 1, "max": 4096}),
            },
        }

    def plan(self, sigmas, explicit_preset, transition_mode, noise_policy="direct_coarse",
             manual_sigma=0.6, delta=0.01, power_A=150.0, power_beta=2.0,
             full_latent_h=45, full_latent_w=80):
        values = [float(s) for s in sigmas]
        scales = SCALE_PRESETS[explicit_preset]
        n_transitions = len(scales) - 1

        base = preset_config(explicit_preset, noise=noise_policy)

        if transition_mode == "manual_sigma":
            # Find sigma boundaries and convert to step indices.
            steps = []
            for idx in range(n_transitions):
                # Each transition uses the same manual_sigma boundary.
                candidates = [i for i, v in enumerate(values[:-1]) if v <= manual_sigma]
                if not candidates:
                    raise ValueError("manual sigma is not reached by the schedule")
                steps.append(candidates[0])
            cfg = base.with_overrides(transition_steps=tuple(steps))
        elif transition_mode == "delta_custom":
            # Delegate to the shared resolution logic.
            config = SpeedConfig(
                scales=tuple(scales),
                transition_steps=base.transition_steps,
                transition_mode="delta_custom",
                noise_policy=noise_policy,
                delta=float(delta),
                power_A=float(power_A),
                power_beta=float(power_beta),
                transition_seed_offset=10_000,
                full_latent_h=int(full_latent_h),
                full_latent_w=int(full_latent_w),
            )
            transition_steps = resolve_transition_steps(config, sigmas)
            cfg = base.with_overrides(
                transition_mode="delta_custom",
                delta=float(delta),
                power_A=float(power_A),
                power_beta=float(power_beta),
                transition_steps=transition_steps,
            )
        else:
            # manual_step: use the preset's default steps unchanged.
            cfg = base

        # Build the human-readable report.
        segs = []
        for idx in range(n_transitions):
            step = int(cfg.transition_steps[idx])
            q = values[step]
            ratio = scales[idx + 1] / scales[idx]
            _, aligned = aligned_speed_sigma(q, ratio)
            segs.append(f"[{scales[idx]}]{step}:{q:.9g}->{aligned:.9g}")
        report = (
            f"preset={explicit_preset} scales={list(scales)} steps={list(cfg.transition_steps)} "
            f"mode={transition_mode} " + " ".join(segs)
        )
        return (cfg, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSchedule": MiniMaxH3SPEEDSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSchedule": "MiniMax H3 SPEED — Schedule"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3SPEEDSchedule"]
