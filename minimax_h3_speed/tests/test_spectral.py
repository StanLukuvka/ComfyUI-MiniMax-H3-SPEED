"""Tests for spectral expansion (spectral.py) — DCT invariants."""

from __future__ import annotations

import sys
from types import ModuleType

import torch


def _install_comfy_stubs():
    comfy = ModuleType("comfy")
    samplers = ModuleType("comfy.samplers")
    utils = ModuleType("comfy.utils")
    model_mgmt = ModuleType("comfy.model_management")
    kdiff = ModuleType("comfy.k_diffusion")
    ksampling = ModuleType("comfy.k_diffusion.sampling")
    nested_tensor = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True
        def __init__(self, tensors):
            self._tensors = tensors
        def unbind(self):
            return self._tensors
    nested_tensor.NestedTensor = NestedTensor

    samplers.sampler_object = lambda name: ("sampler", name)
    utils.PROGRESS_BAR_ENABLED = True

    comfy.samplers = samplers
    comfy.utils = utils
    comfy.model_management = model_mgmt
    comfy.k_diffusion = kdiff
    comfy.k_diffusion.sampling = ksampling
    comfy.nested_tensor = nested_tensor
    sys.modules["comfy"] = comfy
    for name, mod in [("samplers", samplers), ("utils", utils),
                      ("model_management", model_mgmt),
                      ("k_diffusion", kdiff), ("k_diffusion.sampling", ksampling),
                      ("nested_tensor", nested_tensor)]:
        sys.modules["comfy." + name] = mod


_install_comfy_stubs()

import pytest
from minimax_h3_speed.spectral import (
    dct2,
    idct2,
    spectral_expand_dct,
    spectral_expand_dct_coupled,
    dct_temporal,
    idct_temporal,
    spectral_expand_dct_3d,
)


def test_spectral_expand_dct_preserves_low_freq():
    """After expansion, the low-frequency content of the source must be preserved
    (up to DCT rounding error).
    """
    source = torch.randn(1, 4, 16, 16)
    expanded = spectral_expand_dct(source, (32, 32), sigma=0.5, seed=42)
    assert expanded.shape == (1, 4, 32, 32)
    # Low-freq part should match source (inverse DCT of the same low-freq coeffs)
    source_coeffs = dct2(source)
    expanded_coeffs = dct2(expanded)
    # Low-freq block should be identical
    assert torch.allclose(expanded_coeffs[..., :16, :16], source_coeffs, atol=1e-5)


def test_spectral_expand_dct_coupled_preserves_low_freq():
    """Coupled expansion must also preserve source low-freq content."""
    source = torch.randn(1, 4, 16, 16)
    noise = torch.randn(1, 4, 32, 32)
    expanded = spectral_expand_dct_coupled(source, noise, sigma=0.5)
    assert expanded.shape == (1, 4, 32, 32)
    source_coeffs = dct2(source)
    expanded_coeffs = dct2(expanded)
    assert torch.allclose(expanded_coeffs[..., :16, :16], source_coeffs, atol=1e-5)


def test_spectral_expand_dct_sigma_amplitude():
    """High-freq padding amplitude must scale with sigma (the current timestep)."""
    source = torch.zeros(1, 1, 8, 8)
    expanded_01 = spectral_expand_dct(source, (16, 16), sigma=0.1, seed=0)
    expanded_05 = spectral_expand_dct(source, (16, 16), sigma=0.5, seed=0)
    # The high-freq part (outside the 8×8 source) should scale linearly with sigma
    source_h, source_w = 8, 8
    hf_01 = expanded_01[..., source_h:, source_w:].abs().mean()
    hf_05 = expanded_05[..., source_h:, source_w:].abs().mean()
    assert hf_05 > hf_01 * 1.9  # roughly 5x, accounting for randomness


# --- Temporal DCT tests ---

def test_dct_temporal_roundtrip():
    """idct_temporal(dct_temporal(x)) must recover x to within 1e-5."""
    x = torch.randn(1, 4, 8, 16, 16)
    recovered = idct_temporal(dct_temporal(x))
    assert recovered.shape == x.shape
    assert torch.allclose(recovered, x, atol=1e-5)


def test_dct_temporal_dc_component():
    """The DC bin (index 0) of the temporal DCT equals the temporal mean."""
    x = torch.randn(1, 4, 8, 16, 16)
    coeffs = dct_temporal(x)
    # DC = sqrt(1/T) * sum along T; so coeffs[:, :, 0, :, :] should be
    # proportional to the mean along the T axis.
    T = x.shape[-3]
    expected_dc = x.mean(dim=-3) * (T ** 0.5)  # DCT-II scaling
    assert torch.allclose(coeffs[:, :, 0, :, :], expected_dc, atol=1e-5)


def test_spectral_expand_dct_3d_shape():
    """Output shape must match the target_thw tuple."""
    source = torch.randn(1, 4, 4, 8, 8)
    result = spectral_expand_dct_3d(source, (8, 16, 16), sigma=0.5, seed=42)
    assert result.shape == (1, 4, 8, 16, 16)


def test_spectral_expand_dct_3d_lowfreq_preserved():
    """Source's combined temporal+spatial DCT coefficients must appear in the
    low-freq corner of the expanded result."""
    source = torch.randn(1, 4, 4, 8, 8)
    target_t, target_h, target_w = 8, 16, 16
    result = spectral_expand_dct_3d(source, (target_t, target_h, target_w), sigma=0.5, seed=42)

    # Extract the source-shaped low-freq block from the result's combined DCT.
    source_dct = dct2(dct_temporal(source))
    result_dct = dct2(dct_temporal(result))
    source_t, source_h, source_w = source.shape[-3:]
    assert torch.allclose(
        result_dct[..., :source_t, :source_h, :source_w],
        source_dct,
        atol=1e-5,
    )
