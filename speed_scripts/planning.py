"""Stage planning and node-config construction for MiniMax-H3 SPEED."""

from __future__ import annotations

import math

from .config import RATIO_MODES, SpeedConfig


STAGES_TO_SCALES: dict[int, tuple[float, ...]] = {
    2: (0.5, 1.0),
    3: (0.3333333333, 0.6666666667, 1.0),
    4: (0.25, 0.5, 0.75, 1.0),
}


def stage_resolution(
    config: SpeedConfig,
    stage_idx: int,
    full_h: int,
    full_w: int,
    full_t: int,
) -> tuple[int, int, int]:
    """Return ``(h, w, t)`` for one configured stage."""
    scale = config.scales[stage_idx]
    height = max(1, round(full_h * scale))
    width = max(1, round(full_w * scale))
    if config.temporal_scales:
        frames = max(1, round(full_t * config.temporal_scales[stage_idx]))
    else:
        frames = full_t
    return height, width, frames


def power_at_frequency(omega: float, amplitude: float, beta: float) -> float:
    """Radial power-law spectrum ``P(omega) = A * |omega|**(-beta)``."""
    return amplitude * abs(omega) ** (-beta)


def activation_threshold(power: float, delta: float) -> float:
    """Return the SPEED activation threshold for one radial frequency."""
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")
    return 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))


def find_first_step_below(sigmas, threshold: float) -> int:
    """Return the first non-final sigma index at or below `threshold`."""
    values = [float(sigma) for sigma in sigmas]
    last = len(values) - 1
    for index in range(last):
        if values[index] <= threshold:
            return index
    return last


def resolve_transition_steps(
    config: SpeedConfig,
    sigmas,
    H_full: int | None = None,
    W_full: int | None = None,
) -> tuple[int, ...]:
    """Resolve global sigma indices for every resolution transition."""
    if config.transition_mode == "explicit":
        return config.transition_steps
    if H_full is None or W_full is None:
        raise ValueError("delta_custom planning requires live full latent dimensions")

    omega_max = min(H_full, W_full) / 2.0
    steps = []
    for scale in config.scales[:-1]:
        power = power_at_frequency(
            scale * omega_max,
            config.noise_amplitude,
            config.noise_decay_exponent,
        )
        threshold = activation_threshold(power, config.delta)
        steps.append(find_first_step_below(sigmas, threshold))
    return tuple(steps)


def validate_transition_steps(transition_steps, n_sigmas: int) -> None:
    """Validate explicit global boundaries for shared-boundary stage slicing."""
    total_steps = n_sigmas - 1
    if any(not (0 < step < total_steps) for step in transition_steps):
        raise ValueError(
            f"transition goals must be interior step indices "
            f"(0 < goal < {total_steps}): got {list(transition_steps)}"
        )
    if any(left >= right for left, right in zip(transition_steps, transition_steps[1:])):
        raise ValueError(
            f"transition goals must be strictly increasing: got {list(transition_steps)}"
        )


def build_automatic_speed_config(
    *,
    stages,
    noise_policy,
    delta,
    noise_amplitude,
    noise_decay_exponent,
    seed_offset,
) -> SpeedConfig:
    """Build the runtime config used by the Automatic node."""
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=(),
        transition_mode="delta_custom",
        noise_policy=noise_policy,
        delta=float(delta),
        noise_amplitude=float(noise_amplitude),
        noise_decay_exponent=float(noise_decay_exponent),
        transition_seed_offset=int(seed_offset),
    )


def build_manual_speed_config(
    sigmas,
    *,
    transitions,
    ratio_mode: str,
    noise_policy: str,
    seed_offset: int,
) -> SpeedConfig:
    """Normalize Manual-node stage pairs into one explicit ``SpeedConfig``."""
    if ratio_mode not in RATIO_MODES:
        raise ValueError(f"unsupported ratio_mode: {ratio_mode!r}")

    active = [
        (float(goal), float(resolution))
        for goal, resolution in transitions
        if float(goal) != 0.0 and float(resolution) != 0.0
    ]
    if len(active) < 2:
        raise ValueError(
            "manual schedule needs at least two active stages (goal != 0) "
            "with the final stage at resolution 1.0"
        )

    scales = tuple(resolution for _, resolution in active)
    transition_goals = [goal for goal, _ in active[:-1]]
    total_steps = len(sigmas) - 1

    if ratio_mode == "steps":
        for goal in transition_goals:
            if goal != int(goal):
                raise ValueError(
                    f"transition_goal must be a whole step index in steps mode: got {goal}. "
                    "Use ratio_mode='ratio' for fractional (0-1) goals."
                )
        transition_steps = tuple(int(goal) for goal in transition_goals)
    else:
        if any(goal > 1 for goal in transition_goals):
            raise ValueError(
                f"Invalid goal for ratio mode: {transition_goals}. Goals must be <= 1 "
                "(fraction of the schedule)."
            )
        transition_steps = tuple(
            int(round(goal * total_steps)) for goal in transition_goals
        )

    validate_transition_steps(transition_steps, len(sigmas))
    return SpeedConfig(
        scales=scales,
        transition_steps=transition_steps,
        transition_mode="explicit",
        noise_policy=noise_policy,
        delta=0.01,
        transition_seed_offset=int(seed_offset),
    )


__all__ = [
    "STAGES_TO_SCALES",
    "stage_resolution",
    "power_at_frequency",
    "activation_threshold",
    "find_first_step_below",
    "resolve_transition_steps",
    "validate_transition_steps",
    "build_automatic_speed_config",
    "build_manual_speed_config",
]
