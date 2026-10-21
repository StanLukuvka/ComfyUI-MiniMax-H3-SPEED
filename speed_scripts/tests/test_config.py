"""SpeedConfig validation."""

import pytest

from speed_scripts.config import SpeedConfig


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"scales": (1.0,), "transition_steps": ()}, "at least two scales"),
        ({"scales": (.5, .75), "transition_steps": (3,)}, "final scale must be 1.0"),
        ({"scales": (.5, .75, 1.0), "transition_steps": (5,)}, "transition steps"),
        ({"scales": (.5, 1.0), "transition_steps": (5,), "transition_mode": "bad"}, "transition_mode"),
        ({"scales": (.5, 1.0), "transition_steps": (5,), "noise_policy": "bad"}, "noise_policy"),
        ({"scales": (.5, .75, 1.0), "transition_steps": (5, 5)}, "strictly increasing"),
    ],
)
def test_speed_config_rejects_invalid_public_invariants(kwargs, error):
    with pytest.raises(ValueError, match=error):
        SpeedConfig(**kwargs)


@pytest.mark.parametrize("field", ["noise_amplitude", "noise_decay_exponent"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_power_spectrum_parameters_must_be_positive_and_finite(field, value):
    with pytest.raises(ValueError, match="positive finite"):
        SpeedConfig(**{field: value})


def test_temporal_schedule_must_finish_at_full_resolution():
    with pytest.raises(ValueError, match="final temporal scale must be 1.0"):
        SpeedConfig(
            scales=(0.5, 1.0),
            transition_steps=(5,),
            temporal_scales=(0.5, 0.75),
        )


def test_valid_temporal_schedule_can_grow_to_full_resolution():
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),
        temporal_scales=(0.5, 1.0),
    )
    assert config.temporal_scales == (0.5, 1.0)
