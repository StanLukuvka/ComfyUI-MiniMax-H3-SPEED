"""Pure-Torch orthonormal DCT helpers for H3 video spatial axes."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=64)
def _cached_basis(size: int, device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    sample = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequency = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos((math.pi / size) * frequency * sample)
    basis[0] *= math.sqrt(1.0 / size)
    if size > 1:
        basis[1:] *= math.sqrt(2.0 / size)
    return basis


def _basis(size: int, device: torch.device) -> torch.Tensor:
    return _cached_basis(size, device.type, device.index)


def _validate_video_tensor(value: torch.Tensor) -> None:
    if value.ndim < 2:
        raise ValueError("DCT input must have at least two spatial axes")
    if value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError("DCT spatial axes must be non-empty")


def dct2(value: torch.Tensor) -> torch.Tensor:
    _validate_video_tensor(value)
    work = value.float()
    height_basis = _basis(work.shape[-2], work.device)
    width_basis = _basis(work.shape[-1], work.device)
    transformed = torch.matmul(height_basis, work)
    transformed = torch.matmul(transformed, width_basis.transpose(0, 1))
    return transformed


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    _validate_video_tensor(coefficients)
    work = coefficients.float()
    height_basis = _basis(work.shape[-2], work.device)
    width_basis = _basis(work.shape[-1], work.device)
    restored = torch.matmul(height_basis.transpose(0, 1), work)
    restored = torch.matmul(restored, width_basis)
    return restored


def lowpass_dct(value: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    original_dtype = value.dtype
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    source_h, source_w = value.shape[-2:]
    if target_h < 1 or target_w < 1:
        raise ValueError("target spatial axes must be positive")
    if target_h > source_h or target_w > source_w:
        raise ValueError("lowpass target cannot exceed the source shape")
    return idct2(dct2(value)[..., :target_h, :target_w]).to(dtype=original_dtype)


def lowpass_filter_dct(value: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Shape-preserving lowpass filter in the DCT domain.

    Zeroes out coefficients whose frequency index exceeds ``cutoff * full_dim``.
    Returns a tensor with the same shape and dtype as ``value``.
    """
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0, 1]")
    original_dtype = value.dtype
    coeffs = dct2(value.float())
    *leading, H, W = coeffs.shape
    keep_h = max(1, round(H * cutoff))
    keep_w = max(1, round(W * cutoff))
    mask = torch.ones_like(coeffs)
    mask[..., keep_h:, :] = 0.0
    mask[..., :, keep_w:] = 0.0
    return idct2(coeffs * mask).to(dtype=original_dtype)


def spectral_expand_dct_coupled(
    value: torch.Tensor,
    full_resolution_noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    source_h, source_w = value.shape[-2:]
    target_h, target_w = full_resolution_noise.shape[-2:]
    if value.shape[:-2] != full_resolution_noise.shape[:-2]:
        raise ValueError("source state and coupled noise must share leading dimensions")
    if target_h < source_h or target_w < source_w:
        raise ValueError(
            f"DCT cannot expand {(source_h, source_w)} to {(target_h, target_w)}"
        )
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("sigma must be in [0, 1]")
    expanded = dct2(full_resolution_noise).float() * float(sigma)
    expanded[..., :source_h, :source_w] = dct2(value).float()
    return idct2(expanded).to(dtype=value.dtype)


def spectral_expand_dct(
    value: torch.Tensor,
    target_hw: tuple[int, int],
    sigma: float,
    seed: int,
) -> torch.Tensor:
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    source_h, source_w = value.shape[-2:]
    if target_h < source_h or target_w < source_w:
        raise ValueError(
            f"DCT cannot expand {(source_h, source_w)} to {(target_h, target_w)}"
        )
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("sigma must be in [0, 1]")

    source_coefficients = dct2(value).float()
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    expanded = torch.randn(
        value.shape[:-2] + (target_h, target_w),
        generator=generator,
        device=value.device,
        dtype=torch.float32,
    )
    expanded.mul_(float(sigma))
    expanded[..., :source_h, :source_w] = source_coefficients
    return idct2(expanded).to(dtype=value.dtype)


# ---------------------------------------------------------------------------
# Temporal-axis DCT (1D DCT along the T dimension of [B, C, T, H, W] video)
# ---------------------------------------------------------------------------


def dct_temporal(value: torch.Tensor) -> torch.Tensor:
    """1D orthonormal DCT-II along the temporal axis (dim=-3).

    Input shape ``[..., T, H, W]`` (last two dims are spatial; T is at index -3).
    Returns coefficients of the same shape. Uses the same orthonormal
    basis as :func:`dct2` so that ``idct_temporal(dct_temporal(x)) == x``.
    """
    if value.ndim < 3:
        raise ValueError(
            f"dct_temporal expects at least 3 dims (T at axis -3); got {value.ndim}"
        )
    T = value.shape[-3]
    if T < 1:
        raise ValueError("temporal axis must have at least one element")
    work = value.float()
    basis = _basis(T, work.device)  # [T, T]
    # Reshape so the temporal axis is the matrix dimension: [N, T, HW]
    *leading, t_dim, h_dim, w_dim = work.shape
    work_2d = work.reshape(-1, T, h_dim * w_dim)
    transformed = torch.matmul(basis, work_2d)  # [N, T, HW]
    return transformed.reshape(*leading, t_dim, h_dim, w_dim)


def idct_temporal(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse 1D DCT along the temporal axis — the transpose of :func:`dct_temporal`."""
    if coefficients.ndim < 3:
        raise ValueError(
            f"idct_temporal expects at least 3 dims (T at axis -3); got {coefficients.ndim}"
        )
    T = coefficients.shape[-3]
    if T < 1:
        raise ValueError("temporal axis must have at least one element")
    work = coefficients.float()
    basis = _basis(T, work.device)  # [T, T]
    *leading, t_dim, h_dim, w_dim = work.shape
    work_2d = work.reshape(-1, T, h_dim * w_dim)
    restored = torch.matmul(basis.transpose(0, 1), work_2d)  # [N, T, HW]
    return restored.reshape(*leading, t_dim, h_dim, w_dim)


def spectral_expand_dct_3d(
    value: torch.Tensor,           # [..., T_coarse, H_coarse, W_coarse]
    target_thw: tuple[int, int, int],
    sigma: float,
    seed: int,
) -> torch.Tensor:
    """3D spectral expansion: grow temporal AND spatial axes via DCT noise padding.

    Generates full-resolution DCT-domain noise, scales by ``sigma``, embeds the
    source's low-frequency temporal and spatial DCT coefficients into the
    [:source_t, :source_h, :source_w] corner, then inverse-DCTs back to pixel
    space.  This is the natural extension of :func:`spectral_expand_dct` to
    the temporal axis — the temporal spectrum is padded identically.
    """
    target_t, target_h, target_w = (int(target_thw[0]), int(target_thw[1]), int(target_thw[2]))
    if value.ndim < 3:
        raise ValueError("spectral_expand_dct_3d expects at least 3 dims")
    source_t, source_h, source_w = value.shape[-3:]
    if target_t < source_t or target_h < source_h or target_w < source_w:
        raise ValueError(
            f"3D DCT cannot expand {value.shape[-3:]} to {(target_t, target_h, target_w)}"
        )
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("sigma must be in [0, 1]")

    original_dtype = value.dtype
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))

    # Build full-res tensor in the *combined* DCT domain (temporal + spatial).
    # The composition order doesn't matter because the two DCTs operate on
    # disjoint axes (T vs. (H, W)); we apply temporal first, then spatial.
    target_shape = value.shape[:-3] + (target_t, target_h, target_w)
    dct_full = torch.randn(
        target_shape, generator=generator,
        device=value.device, dtype=torch.float32,
    )
    dct_full.mul_(float(sigma))

    # Source's combined temporal+spatial DCT coefficients (same shape as value).
    source_dct = dct2(dct_temporal(value))
    dct_full[..., :source_t, :source_h, :source_w] = source_dct

    # Inverse DCT back to pixel domain. idct2 then idct_temporal because the
    # composition is separable and commutative.
    return idct_temporal(idct2(dct_full)).to(dtype=original_dtype)


__all__ = ["dct2", "idct2", "lowpass_dct", "lowpass_filter_dct",
           "spectral_expand_dct", "spectral_expand_dct_coupled",
           "dct_temporal", "idct_temporal", "spectral_expand_dct_3d"]
