# FLOW-PRODUCED — Implementation Plan — Continuous SPEED Sigma Harvester.md §7 (commit 1) — flow-produced, do not hand-edit
"""Shared automatic SPEED config construction.

The Automatic SPEED sampler and the SPEED Sigma Harvest diagnostic node must
build identical SpeedConfig values from the same widget inputs. This module
owns that construction so the two nodes cannot drift: the stage->scale
ladder and the SpeedConfig fields all live here, and both nodes call
`build_automatic_speed_config`. The delta value arrives already
alias-resolved — the widget-label alias handling stays in the node layer.
"""

from __future__ import annotations

from speed_scripts.config import SpeedConfig
from speed_scripts.nodes_common import full_res_dims


# Stages -> scale ladder for Automatic. Evenly spaced, ends at 1.0.
# 2: 0.5 → 1.0, 3: 0.33 → 0.66 → 1.0, 4: 0.25 → 0.5 → 0.75 → 1.0
STAGES_TO_SCALES: dict[int, tuple[float, ...]] = {
    2: (0.5, 1.0),
    3: (0.3333333333, 0.6666666667, 1.0),
    4: (0.25, 0.5, 0.75, 1.0),
}
# Backwards compat: old preset names -> stages (for workflows saved before the rename)
PRESET_TO_STAGES: dict[str, int] = {
    "half_then_full": 2,
    "three_quarter_then_full": 2,
    "quarter_half_full": 3,
    "aggressive": 3,
    "quarter_half_3q_full": 4,
}


def build_automatic_speed_config(
    latent_image,
    *,
    stages,
    noise_policy,
    delta,
    noise_amplitude,
    noise_decay_exponent,
    seed_offset,
) -> SpeedConfig:
    """Build the SpeedConfig the Automatic sampler runs today.

    `stages` selects the scale ladder and must already be normalized to
    2-4 (the caller clamps). The transition steps are dummies: in
    delta_custom mode the runtime places the real boundaries from the
    power-spectrum threshold, so only the count matters (one boundary
    between each pair of stages). `delta` is the Tolerance (Delta)
    widget value.
    """
    scales = STAGES_TO_SCALES[stages]
    transition_steps = tuple(range(1, len(scales)))
    full_h, full_w = full_res_dims(latent_image)
    return SpeedConfig(
        scales=tuple(scales),
        transition_steps=transition_steps,
        transition_mode="delta_custom",
        noise_policy=noise_policy,
        delta=float(delta),
        noise_amplitude=float(noise_amplitude),
        noise_decay_exponent=float(noise_decay_exponent),
        transition_seed_offset=int(seed_offset),
        full_latent_h=full_h,
        full_latent_w=full_w,
    )


__all__ = [
    "STAGES_TO_SCALES",
    "PRESET_TO_STAGES",
    "build_automatic_speed_config",
]
