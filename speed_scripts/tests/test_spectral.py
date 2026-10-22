"""direct_coarse transition noise: the carried band is kept, new bands get transition-seeded noise."""

import torch

from speed_scripts.spectral import dct2, spectral_expand


def test_spectral_expand_keeps_the_source_band_and_seeds_the_new_bands():
    source = torch.randn(1, 2, 4, 8, 12)
    first = spectral_expand(source, (16, 24), sigma=0.5, seed=99)

    assert torch.allclose(dct2(first)[..., :8, :12], dct2(source), atol=1e-5)
    assert torch.equal(first, spectral_expand(source, (16, 24), sigma=0.5, seed=99))
    assert not torch.equal(first, spectral_expand(source, (16, 24), sigma=0.5, seed=100))

    # The new-band noise amplitude is the transition sigma.
    zero = torch.zeros(1, 1, 8, 8)
    low, high = (dct2(spectral_expand(zero, (16, 16), sigma=s, seed=3)) for s in (0.1, 0.5))
    low[..., :8, :8] = high[..., :8, :8] = 0
    assert torch.allclose(high, low * 5.0, atol=1e-5, rtol=1e-5)
