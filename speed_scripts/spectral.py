"""Pure-Torch orthonormal DCT helpers for H3 video latents."""

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


def _validate_spatial_tensor(value: torch.Tensor) -> None:
    if value.ndim < 2:
        raise ValueError("DCT input must have at least two spatial axes")
    if value.shape[-2] < 1 or value.shape[-1] < 1:
        raise ValueError("DCT spatial axes must be non-empty")


def dct2(value: torch.Tensor) -> torch.Tensor:
    _validate_spatial_tensor(value)
    work = value.float()
    height_basis = _basis(work.shape[-2], work.device)
    width_basis = _basis(work.shape[-1], work.device)
    transformed = torch.matmul(height_basis, work)
    transformed = torch.matmul(transformed, width_basis.transpose(0, 1))
    return transformed


def idct2(coefficients: torch.Tensor) -> torch.Tensor:
    _validate_spatial_tensor(coefficients)
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


def spectral_expand(
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


def dct_temporal(value: torch.Tensor) -> torch.Tensor:
    """1D orthonormal DCT-II along the temporal axis (dim=-3)."""
    if value.ndim < 3:
        raise ValueError(
            f"dct_temporal expects at least 3 dims (T at axis -3); got {value.ndim}"
        )
    T = value.shape[-3]
    if T < 1:
        raise ValueError("temporal axis must have at least one element")
    work = value.float()
    basis = _basis(T, work.device)
    *leading, t_dim, h_dim, w_dim = work.shape
    work_2d = work.reshape(-1, T, h_dim * w_dim)
    transformed = torch.matmul(basis, work_2d)
    return transformed.reshape(*leading, t_dim, h_dim, w_dim)


def idct_temporal(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse 1D DCT along the temporal axis."""
    if coefficients.ndim < 3:
        raise ValueError(
            f"idct_temporal expects at least 3 dims (T at axis -3); got {coefficients.ndim}"
        )
    T = coefficients.shape[-3]
    if T < 1:
        raise ValueError("temporal axis must have at least one element")
    work = coefficients.float()
    basis = _basis(T, work.device)
    *leading, t_dim, h_dim, w_dim = work.shape
    work_2d = work.reshape(-1, T, h_dim * w_dim)
    restored = torch.matmul(basis.transpose(0, 1), work_2d)
    return restored.reshape(*leading, t_dim, h_dim, w_dim)


def spectral_expand_3d(
    value: torch.Tensor,
    target_thw: tuple[int, int, int],
    sigma: float,
    seed: int,
) -> torch.Tensor:
    """Grow temporal and spatial axes by padding the combined DCT domain with noise."""
    target_t, target_h, target_w = (
        int(target_thw[0]),
        int(target_thw[1]),
        int(target_thw[2]),
    )
    if value.ndim < 3:
        raise ValueError("spectral_expand_3d expects at least 3 dims")
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
    target_shape = value.shape[:-3] + (target_t, target_h, target_w)
    dct_full = torch.randn(
        target_shape,
        generator=generator,
        device=value.device,
        dtype=torch.float32,
    )
    dct_full.mul_(float(sigma))

    source_dct = dct2(dct_temporal(value))
    dct_full[..., :source_t, :source_h, :source_w] = source_dct
    return idct_temporal(idct2(dct_full)).to(dtype=original_dtype)


__all__ = [
    "dct2",
    "idct2",
    "lowpass_dct",
    "spectral_expand",
    "dct_temporal",
    "idct_temporal",
    "spectral_expand_3d",
]
