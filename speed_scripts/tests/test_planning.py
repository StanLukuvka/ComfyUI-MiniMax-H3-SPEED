"""Stage ladders, transition-threshold math, and Automatic config building."""

import math

import pytest

from conftest import SIGMAS
from speed_scripts.config import SpeedConfig
from speed_scripts.planning import (
    STAGES_TO_SCALES,
    activation_threshold,
    build_automatic_speed_config,
    power_at_frequency,
    resolve_transition_steps,
)


def _automatic_config(stages):
    return build_automatic_speed_config(
        stages=stages,
        noise_policy="direct_coarse",
        delta=0.005,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
    )


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_builder_maps_stage_count_to_exact_scale_ladder(stages):
    cfg = _automatic_config(stages)
    assert cfg.scales == STAGES_TO_SCALES[stages]
    assert cfg.transition_steps == ()
    assert cfg.transition_mode == "delta_custom"


@pytest.mark.parametrize("stages", [1, 5, True, 3.0, "3"])
def test_builder_rejects_invalid_stage_counts(stages):
    with pytest.raises(ValueError, match="stages must be exactly 2, 3, or 4"):
        _automatic_config(stages)


def test_power_spectrum_activation_math_matches_reference_equations():
    omega, amplitude, beta, delta = 8.0, 12.5, 1.8, .01
    power = power_at_frequency(omega, amplitude, beta)
    assert power == pytest.approx(amplitude * abs(omega) ** (-beta))
    expected = 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))
    assert activation_threshold(power, delta) == pytest.approx(expected)


def test_delta_custom_planning_requires_live_geometry():
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(),
        transition_mode="delta_custom",
    )
    with pytest.raises(ValueError, match="live full latent dimensions"):
        resolve_transition_steps(cfg, SIGMAS)


def test_explicit_planning_does_not_require_geometry():
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
    )
    assert resolve_transition_steps(cfg, SIGMAS) == (5,)
