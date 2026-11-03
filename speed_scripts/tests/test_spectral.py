"""direct_coarse transition noise: the carried band is kept, new bands get transition-seeded noise."""

import pytest
import torch

from speed_scripts.spectral import dct2, spectral_expand, spectral_expand_3d


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_transition_fill_is_the_same_on_every_device():
    source = torch.randn(1, 2, 4, 8, 12)
    for expand, target in ((spectral_expand, (16, 24)), (spectral_expand_3d, (6, 16, 24))):
        on_cpu = expand(source, target, sigma=0.5, seed=10291)
        on_gpu = expand(source.cuda(), target, sigma=0.5, seed=10291)
        assert on_gpu.device.type == "cuda"
        assert torch.allclose(on_gpu.cpu(), on_cpu, atol=1e-5)
