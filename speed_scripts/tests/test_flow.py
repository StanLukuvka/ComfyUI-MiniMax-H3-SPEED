"""Tests for the flow module (flow.py) — kappa, aligned sigma, reentry noise."""

from __future__ import annotations

import torch

# One canonical comfy stub installer lives in conftest; import under the old
# name so the single call site below stays valid.
from conftest import install_comfy_stubs as _install_comfy_stubs


_install_comfy_stubs()

import pytest
from speed_scripts.flow import aligned_sigma, reentry_noise


def test_kappa_formula():
    """κ = r / (1 + (r-1)t) per Eq. (5)."""
    r = 2.0
    t = 0.5
    kappa, new_q = aligned_sigma(t, r)
    expected_kappa = r / (1.0 + (r - 1.0) * t)
    assert abs(kappa - expected_kappa) < 1e-10


def test_aligned_sigma_formula():
    """t̃ = κ * t per Eq. (6) (= r·t / (1 + (r-1)t))."""
    r = 2.0
    t = 0.5
    kappa, new_q = aligned_sigma(t, r)
    expected_new_q = t * kappa
    assert abs(new_q - expected_new_q) < 1e-10


def test_aligned_sigma_boundary_values():
    """At t→0, κ→r and t̃→r·t (noise-dominated regime).
    At t→1, κ→1 and t̃→1 (signal-dominated regime, no rescaling needed).
    """
    r = 2.0
    # Small t: κ should approach r
    kappa, _ = aligned_sigma(0.001, r)
    assert kappa > r * 0.99
    # Large t (close to 1): κ should approach 1
    kappa, _ = aligned_sigma(0.999, r)
    assert abs(kappa - 1.0) < 0.01


def test_aligned_sigma_invalid_inputs():
    """Must reject q <= 0, q >= 1, r <= 1."""
    with pytest.raises(ValueError):
        aligned_sigma(0.0, 2.0)
    with pytest.raises(ValueError):
        aligned_sigma(1.0, 2.0)
    with pytest.raises(ValueError):
        aligned_sigma(0.5, 1.0)
    with pytest.raises(ValueError):
        aligned_sigma(0.5, 0.5)


def test_reentry_noise_formula():
    """reentry_noise(internal, start_sigma) = internal / start_sigma."""
    internal = torch.tensor([1.0, 2.0, 3.0])
    sigma = 0.5
    result = reentry_noise(internal, sigma)
    expected = internal / sigma
    assert torch.allclose(result, expected)


def test_reentry_noise_raises_on_zero():
    with pytest.raises(ValueError, match="start_sigma must be positive"):
        reentry_noise(torch.zeros(3), 0.0)
