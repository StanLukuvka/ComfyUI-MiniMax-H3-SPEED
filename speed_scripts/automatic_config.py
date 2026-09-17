"""Automatic SPEED stage ladder and config construction."""

from __future__ import annotations

from speed_scripts.config import SpeedConfig
from speed_scripts.nodes_common import full_res_dims


STAGES_TO_SCALES: dict[int, tuple[float, ...]] = {
    2: (0.5, 1.0),
    3: (0.3333333333, 0.6666666667, 1.0),
    4: (0.25, 0.5, 0.75, 1.0),
}

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
    """Build the runtime config used by the Automatic node."""
    full_h, full_w = full_res_dims(latent_image)
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=(),
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
