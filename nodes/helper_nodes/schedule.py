"""Schedule node — computes SpeedConfig from sigmas + preset + mode."""

from __future__ import annotations

import math

import torch

from minimax_h3_speed.config import SCALE_PRESETS, preset_config, SpeedConfig
from minimax_h3_speed.h3_runtime import (
    _find_first_step_below,
    power_spectrum,
    activation_time,
)


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
                "preset": (list(SCALE_PRESETS.keys()),),
                "transition_mode": (["manual_step", "manual_sigma", "delta_custom"],),
                "manual_sigma": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
                "delta": ("FLOAT", {"default": 0.01, "min": 0.000001, "max": 0.999999, "step": 0.001}),
                "power_A": ("FLOAT", {"default": 219.48, "min": 0.000001, "max": 1000000.0}),
                "power_beta": ("FLOAT", {"default": 2.42, "min": 0.000001, "max": 10.0}),
                "full_latent_h": ("INT", {"default": 45, "min": 1, "max": 4096}),
                "full_latent_w": ("INT", {"default": 80, "min": 1, "max": 4096}),
            },
        }

    def plan(self, sigmas, preset, transition_mode,
             manual_sigma=0.6, delta=0.01, power_A=219.48, power_beta=2.42,
             full_latent_h=45, full_latent_w=80):
        values = [float(s) for s in sigmas]
        scales = SCALE_PRESETS[preset]
        n_transitions = len(scales) - 1

        # Start from the calibrated default transition steps for this preset.
        base = preset_config(preset)
        transition_steps = list(base.transition_steps)

        if transition_mode == "manual_sigma":
            for idx in range(n_transitions):
                candidates = [i for i, value in enumerate(values[:-1]) if value <= manual_sigma]
                if not candidates:
                    raise ValueError("manual sigma is not reached by the schedule")
                transition_steps[idx] = candidates[0]
        elif transition_mode == "delta_custom":
            for idx in range(n_transitions):
                scale = scales[idx]
                omega = scale * min(full_latent_h, full_latent_w) / 2.0
                power = power_A * abs(omega) ** (-power_beta)
                threshold = 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))
                candidates = [i for i, value in enumerate(values[:-1]) if value <= threshold]
                if not candidates:
                    raise ValueError("custom delta threshold is not reached by the schedule")
                transition_steps[idx] = candidates[0]
        # manual_step uses the base default steps unchanged.

        cfg = base.with_overrides(transition_steps=tuple(transition_steps))
        segs = []
        for idx in range(n_transitions):
            q = values[int(transition_steps[idx])]
            ratio = scales[idx + 1] / scales[idx]
            from minimax_h3_speed.flow import aligned_speed_sigma
            _, aligned = aligned_speed_sigma(q, ratio)
            segs.append(f"[{scales[idx]}]{int(transition_steps[idx])}:{q:.9g}->{aligned:.9g}")
        report = (
            f"preset={preset} scales={list(scales)} steps={list(transition_steps)} "
            f"mode={transition_mode} " + " ".join(segs)
        )
        return (cfg, report)


NODE_CLASS_MAPPINGS = {"MiniMaxH3SPEEDSchedule": MiniMaxH3SPEEDSchedule}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3SPEEDSchedule": "MiniMax H3 SPEED — Schedule"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "MiniMaxH3SPEEDSchedule"]
