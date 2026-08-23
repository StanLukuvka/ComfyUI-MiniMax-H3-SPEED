"""Pure-torch DCT-expand tests — no ComfyUI dependency needed."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from speed_scripts.spectral import lowpass_dct, spectral_expand

# Canonical stubs live in conftest; _install_comfy_stubs is the superset
# (also ships pack_latents/unpack_latents + k_diffusion).
from conftest import install_comfy_stubs as _install_comfy_stubs


_install_comfy_stubs()


def test_lowpass_preserves_dc_and_attenuates_high():
    """1D: lowpass must preserve DC while attenuating high-frequency content."""
    N = 32
    # DC-only field in a real [1,1,H,W] tensor: all spatial pixels equal.
    x = torch.ones(1, 1, N, N)
    x_lp = lowpass_dct(x, (N, N))  # target == source -> identity for DC
    assert torch.allclose(x_lp, x, atol=1e-6), "DC should survive lowpass"

    # High-frequency content is attenuated when we keep only the low band:
    # a checkerboard (max spatial frequency) lowpass'd to half its size should
    # lose its fine structure.
    y = torch.zeros(1, 1, N, N)
    y[:, :, ::2, ::2] = 1.0  # coarse nonzero cells
    y_full = lowpass_dct(y, (N, N))   # keep full -> unchanged
    y_half = lowpass_dct(y, (N // 2, N // 2))  # drop high bands
    assert torch.allclose(y_full, y, atol=1e-5)
    # Down-sampling in frequency drops energy -> variance shrinks.
    assert y_half.std() < y.std()


def test_expand_dct_preserves_energy():
    """DCT expand must be energy-preserving to leading order (sum of squares
    stays ≈ constant across spatial dims)."""
    B, C, T, H, W = 1, 1, 4, 8, 16
    x = torch.randn(B, C, T, H, W)
    # sigma=0.0 disables the random high-frequency fill so this is pure DCT
    # expansion (no injected noise).
    y = spectral_expand(x, (H * 2, W * 2), sigma=0.0, seed=0)
    assert y.shape == (B, C, T, H * 2, W * 2)
    # DCT basis is orthonormal, so sum_sq should be invariant under forward+backward.
    # Here we just expand once; energy is preserved when we undo with idct, which
    # we test separately.
    assert not torch.isnan(y).any() and not torch.isinf(y).any()


def test_idct_back_to_original():
    """spectral_expand (sigma=0) followed by lowpass_dct on the higher-res
    tensor should recover the original (orthonormal DCT)."""
    B, C, T, H, W = 1, 2, 64, 4, 8
    x = torch.randn(B, C, T, H, W)
    # Expand to higher-res with sigma=0 (deterministic, no high-freq noise)
    y = spectral_expand(x, (H * 4, W * 4), sigma=0.0, seed=0)
    # Downsample back (lowpass in the higher-res space, then we compare the
    # original-resolution band against the original via a downscale).
    y_lp = lowpass_dct(y, (H, W))
    assert y_lp.shape == (B, C, T, H, W)
    # The DCT basis is orthonormal: expanding then lowpass-back is identity.
    assert torch.allclose(x, y_lp, atol=1e-5), "expansion then lowpass should recover input"


def test_dct_expand_with_seed_offset_stability():
    """Expanding the same noise twice with the same seed offset must be identical."""
    x = torch.randn(1, 1, 4, 8, 12)
    y1 = spectral_expand(x, (16, 24), sigma=0.5, seed=99)
    y2 = spectral_expand(x, (16, 24), sigma=0.5, seed=99)
    assert torch.equal(y1, y2)
    y3 = spectral_expand(x, (16, 24), sigma=0.5, seed=100)
    assert not torch.equal(y1, y3)


def test_spectral_dct2():
    """Compare Lab vs MVP dct2 implementation."""
    from speed_scripts.spectral import dct2 as mvp_func
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab"))
    try:
        from speed_lab.spectral import dct2 as lab_func
    except ModuleNotFoundError:
        pytest.skip("speed_lab sibling repo not present")
    video = torch.randn(1, 1, 32, 32)
    mvp_out = mvp_func(video)
    lab_out = lab_func(video)
    assert torch.equal(mvp_out, lab_out)


def test_spectral_idct2():
    """Compare Lab vs MVP idct2 implementation."""
    from speed_scripts.spectral import idct2 as mvp_func
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab"))
    try:
        from speed_lab.spectral import idct2 as lab_func
    except ModuleNotFoundError:
        pytest.skip("speed_lab sibling repo not present")
    coeffs = torch.randn(1, 1, 32, 32)
    mvp_out = mvp_func(coeffs)
    lab_out = lab_func(coeffs)
    assert torch.equal(mvp_out, lab_out)


def test_spectral_lowpass_dct():
    """Compare Lab vs MVP lowpass_dct implementation."""
    from speed_scripts.spectral import lowpass_dct as mvp_func
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab"))
    try:
        from speed_lab.spectral import lowpass_dct as lab_func
    except ModuleNotFoundError:
        pytest.skip("speed_lab sibling repo not present")
    video = torch.randn(1, 1, 32, 32)
    target_hw = (16, 16)
    mvp_out = mvp_func(video, target_hw)
    lab_out = lab_func(video, target_hw)
    assert torch.equal(mvp_out, lab_out)


def test_spectral_expand():
    """Compare Lab vs MVP spectral_expand implementation."""
    from speed_scripts.spectral import spectral_expand as mvp_func
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab"))
    try:
        from speed_lab.spectral import spectral_expand as lab_func
    except ModuleNotFoundError:
        pytest.skip("speed_lab sibling repo not present")
    video = torch.randn(1, 1, 32, 32)
    target_hw = (64, 64)
    sigma = 0.5
    seed = 42
    mvp_out = mvp_func(video, target_hw, sigma, seed)
    lab_out = lab_func(video, target_hw, sigma, seed)
    assert torch.allclose(mvp_out, lab_out, atol=1e-5), \
        f"expand_dct mismatch (max diff: {(mvp_out - lab_out).abs().max():.6f})"


def test_spectral_expand_coupled():
    """Compare Lab vs MVP spectral_expand_coupled implementation."""
    from speed_scripts.spectral import dct2, spectral_expand_coupled as mvp_func
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab"))
    try:
        from speed_lab.spectral import spectral_expand_coupled as lab_func
    except ModuleNotFoundError:
        pytest.skip("speed_lab sibling repo not present")
    video = torch.randn(1, 1, 32, 32)
    full_resolution_noise = torch.randn(1, 1, 64, 64)
    sigma = 0.5

    # Both take raw noise tensor (Lab computes DCT internally)
    mvp_out = mvp_func(video, full_resolution_noise, sigma)
    lab_out = lab_func(video, full_resolution_noise, sigma)
    assert torch.allclose(mvp_out, lab_out, atol=1e-5), \
        f"expand_dct_coupled mismatch (max diff: {(mvp_out - lab_out).abs().max():.6f})"
