"""Core flow-matching math contracts."""

import pytest
import torch

from speed_scripts.flow import aligned_sigma, reentry_noise, time_shift_sigma


@pytest.mark.parametrize("q,ratio", [(0.5, 2.0), (0.3, 2.0), (0.8, 4.0 / 3.0)])
def test_aligned_sigma_matches_speed_equations(q, ratio):
    kappa, aligned = aligned_sigma(q, ratio)
    expected_kappa = ratio / (1.0 + (ratio - 1.0) * q)
    assert kappa == pytest.approx(expected_kappa)
    assert aligned == pytest.approx(q * expected_kappa)


@pytest.mark.parametrize(
    "q,ratio",
    [(0.0, 2.0), (1.0, 2.0), (0.5, 1.0), (0.5, 0.5)],
)
def test_aligned_sigma_rejects_invalid_domain(q, ratio):
    with pytest.raises(ValueError):
        aligned_sigma(q, ratio)


def test_reentry_noise_round_trip_and_zero_guard():
    internal = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(reentry_noise(internal, 0.5), internal / 0.5)
    with pytest.raises(ValueError, match="start_sigma must be positive"):
        reentry_noise(internal, 0.0)


def test_time_shift_sigma_maps_between_video_and_audio_clocks():
    assert time_shift_sigma(0.5, 2.0, 1.0) == pytest.approx(1.0 / 3.0)
    assert time_shift_sigma(0.5, 1.0, 1.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        time_shift_sigma(0.5, 0.0, 1.0)
    with pytest.raises(ValueError):
        time_shift_sigma(0.5, 1.0, -1.0)
