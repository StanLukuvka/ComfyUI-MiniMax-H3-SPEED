"""Tests for spectral expansion (spectral.py) — DCT invariants."""

from __future__ import annotations

import torch

from conftest import install_comfy_stubs as _install_comfy_stubs


_install_comfy_stubs()

import pytest
from speed_scripts.spectral import (
    dct2,
    idct2,
    spectral_expand,
    spectral_expand_coupled,
)


def test_spectral_expand_preserves_low_freq():
    """After expansion, the low-frequency content of the source must be preserved
    (up to DCT rounding error).
    """
    source = torch.randn(1, 4, 16, 16)
    expanded = spectral_expand(source, (32, 32), sigma=0.5, seed=42)
    assert expanded.shape == (1, 4, 32, 32)
    # Low-freq part should match source (inverse DCT of the same low-freq coeffs)
    source_coeffs = dct2(source)
    expanded_coeffs = dct2(expanded)
    # Low-freq block should be identical
    assert torch.allclose(expanded_coeffs[..., :16, :16], source_coeffs, atol=1e-5)


def test_spectral_expand_coupled_preserves_low_freq():
    """Coupled expansion must also preserve source low-freq content."""
    source = torch.randn(1, 4, 16, 16)
    noise = torch.randn(1, 4, 32, 32)
    expanded = spectral_expand_coupled(source, noise, sigma=0.5)
    assert expanded.shape == (1, 4, 32, 32)
    source_coeffs = dct2(source)
    expanded_coeffs = dct2(expanded)
    assert torch.allclose(expanded_coeffs[..., :16, :16], source_coeffs, atol=1e-5)


def test_spectral_expand_sigma_amplitude():
    """High-freq padding amplitude must scale with sigma (the current timestep)."""
    source = torch.zeros(1, 1, 8, 8)
    expanded_01 = spectral_expand(source, (16, 16), sigma=0.1, seed=0)
    expanded_05 = spectral_expand(source, (16, 16), sigma=0.5, seed=0)
    # The high-freq part (outside the 8×8 source) should scale linearly with sigma
    source_h, source_w = 8, 8
    hf_01 = expanded_01[..., source_h:, source_w:].abs().mean()
    hf_05 = expanded_05[..., source_h:, source_w:].abs().mean()
    assert hf_05 > hf_01 * 1.9  # roughly 5x, accounting for randomness
