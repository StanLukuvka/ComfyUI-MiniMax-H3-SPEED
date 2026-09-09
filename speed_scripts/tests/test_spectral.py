"""Spectral transform and expansion contracts."""

import pytest
import torch

from speed_scripts.spectral import (
    dct2,
    idct2,
    lowpass_dct,
    spectral_expand,
    spectral_expand_coupled,
)


def test_dct_round_trip_and_lowpass_recovery():
    x = torch.randn(1, 2, 4, 8, 16)
    assert torch.allclose(idct2(dct2(x)), x, atol=1e-5)

    expanded = spectral_expand(x, (16, 32), sigma=0.0, seed=7)
    recovered = lowpass_dct(expanded, (8, 16))
    assert torch.allclose(recovered, x, atol=1e-5)


def test_lowpass_preserves_dc_and_removes_fine_structure():
    dc = torch.ones(1, 1, 32, 32)
    assert torch.allclose(lowpass_dct(dc, (32, 32)), dc, atol=1e-6)

    checker = torch.zeros(1, 1, 32, 32)
    checker[..., ::2, ::2] = 1.0
    coarse = lowpass_dct(checker, (16, 16))
    assert coarse.std() < checker.std()


def test_spectral_expand_preserves_source_band_and_seed_contract():
    source = torch.randn(1, 2, 4, 8, 12)
    first = spectral_expand(source, (16, 24), sigma=0.5, seed=99)
    same = spectral_expand(source, (16, 24), sigma=0.5, seed=99)
    different = spectral_expand(source, (16, 24), sigma=0.5, seed=100)

    assert torch.equal(first, same)
    assert not torch.equal(first, different)
    source_coeffs = dct2(source)
    expanded_coeffs = dct2(first)
    assert torch.allclose(expanded_coeffs[..., :8, :12], source_coeffs, atol=1e-5)


def test_spectral_expand_noise_amplitude_tracks_sigma():
    source = torch.zeros(1, 1, 8, 8)
    low = dct2(spectral_expand(source, (16, 16), sigma=0.1, seed=3))
    high = dct2(spectral_expand(source, (16, 16), sigma=0.5, seed=3))
    low[..., :8, :8] = 0
    high[..., :8, :8] = 0
    assert torch.allclose(high, low * 5.0, atol=1e-5, rtol=1e-5)


def test_coupled_expand_preserves_source_band():
    source = torch.randn(1, 2, 8, 8)
    noise = torch.randn(1, 2, 16, 16)
    expanded = spectral_expand_coupled(source, noise, sigma=0.5)
    assert expanded.shape == noise.shape
    assert torch.allclose(
        dct2(expanded)[..., :8, :8],
        dct2(source),
        atol=1e-5,
    )
