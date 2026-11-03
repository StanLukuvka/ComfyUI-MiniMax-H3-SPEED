"""Seed-portable noise: video noise keyed by frequency index, audio noise on its own generator."""

import torch

from speed_scripts.runtime import (
    _seeded_randn,
    portable_coefficients,
    portable_noise,
    split_extra_noise,
)
from speed_scripts.spectral import dct2, lowpass_dct


def _noise(h, w, seed=1):
    coefficients = portable_coefficients((1, 4, 3, h, w), seed)
    return portable_noise(coefficients, (3, h, w))


def test_portable_noise_is_unit_gaussian():
    x = _noise(40, 72)
    assert abs(float(x.mean())) < 0.02
    assert abs(float(x.std()) - 1.0) < 0.02


def test_neighbouring_grids_share_low_frequencies():
    a, b = _noise(8, 14), _noise(8, 15)
    shared = dct2(a)[..., :8, :14], dct2(b)[..., :8, :14]
    assert torch.allclose(*shared, atol=1e-5)
    other = dct2(_noise(8, 15, seed=2))[..., :8, :14]
    corr = torch.corrcoef(torch.stack([shared[0].ravel(), other.ravel()]))[0, 1]
    assert abs(float(corr)) < 0.1


def test_old_policy_noise_is_unrelated_across_shapes():
    a = _seeded_randn((1, 4, 3, 8, 14), 1)
    b = lowpass_dct(_seeded_randn((1, 4, 3, 8, 15), 1), (8, 14))
    corr = torch.corrcoef(torch.stack([a.ravel(), b.ravel()]))[0, 1]
    assert abs(float(corr)) < 0.1


def test_split_extra_noise_is_seeded_and_skips_a_missing_stream():
    audio = torch.zeros((1, 32, 2, 50))
    assert torch.equal(split_extra_noise(audio, 3), split_extra_noise(audio, 3))
    assert not torch.equal(split_extra_noise(audio, 3), split_extra_noise(audio, 4))
    assert split_extra_noise(None, 3) is None


def test_portable_coefficients_reject_oversized_latents():
    import pytest

    with pytest.raises(ValueError, match="portable_spectrum"):
        portable_coefficients((1, 4, 1, 8, 300), 1)
