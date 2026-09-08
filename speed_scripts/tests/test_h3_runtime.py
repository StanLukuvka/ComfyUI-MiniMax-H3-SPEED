"""Focused H3 runtime/config safety contracts."""

import pytest

from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import resolve_sigma_shifts


def test_h3_sigma_shifts_win_over_generic_comfy_sampling_shift():
    generic = type("Sampling", (), {"shift": 1.0, "audio_shift": 1.0})()

    class Patcher:
        model = type(
            "Model",
            (),
            {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0},
        )()

        def get_model_object(self, name):
            return generic if name == "model_sampling" else None

    guider = type("Guider", (), {"model_patcher": Patcher()})()
    video, audio, scale = resolve_sigma_shifts(guider)
    assert (video, audio, scale) == pytest.approx((12.0, 3.0, 4.0))


def test_non_h3_model_is_rejected_instead_of_using_generic_shift():
    class Patcher:
        model = object()

        def get_model_object(self, name):
            return type("Sampling", (), {"shift": 1.0, "audio_shift": 1.0})()

    guider = type("Guider", (), {"model_patcher": Patcher()})()
    with pytest.raises(ValueError, match="MiniMax-H3 sigma shifts are unavailable"):
        resolve_sigma_shifts(guider)


def test_power_spectrum_activation_math_matches_reference_equations():
    import math
    from speed_scripts.h3_runtime import activation_threshold, power_at_frequency

    omega, amplitude, beta, delta = 8.0, 12.5, 1.8, .01
    power = power_at_frequency(omega, amplitude, beta)
    assert power == pytest.approx(amplitude * abs(omega) ** (-beta))
    expected = 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))
    assert activation_threshold(power, delta) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"scales": (.5,), "transition_steps": ()}, "single scale must be 1.0"),
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
