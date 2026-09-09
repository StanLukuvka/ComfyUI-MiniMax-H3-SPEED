"""Torch-native online spectral-analysis contracts."""

import json
import math

import numpy as np
import pytest
import torch

from speed_scripts.harvest import fit_power_law, radial_dct_power
from speed_scripts.online_harvest import (
    build_fit_record,
    dumps_strict,
    fit_power_law_torch,
    radial_dct_power_torch,
    sample_radial_band_power,
    sample_radial_power,
    update_log_ema,
)


def _basis_row(k, n):
    x = torch.arange(n, dtype=torch.float32) + .5
    scale = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
    return torch.cos(math.pi / n * k * x) * scale


def test_radial_power_identifies_frequency_and_matches_numpy_path():
    basis = torch.einsum("i,j->ij", _basis_row(3, 16), _basis_row(0, 16))
    freqs, profile = radial_dct_power_torch(basis[None, None, None])
    assert freqs[int(profile.argmax())] == 3

    torch.manual_seed(7)
    video = torch.randn(2, 3, 2, 16, 24)
    freqs_np, profile_np = radial_dct_power(video)
    freqs_t, profile_t = radial_dct_power_torch(video)
    assert np.array_equal(freqs_np, freqs_t.cpu().numpy())
    assert np.allclose(profile_np, profile_t.cpu().numpy(), rtol=1e-4, atol=1e-7)


def test_radial_power_rejects_non_video_tensor():
    with pytest.raises(ValueError):
        radial_dct_power_torch(torch.zeros(4, 16, 16))


def test_power_law_fit_recovers_signal_and_matches_numpy_fitter():
    omega = torch.arange(1.0, 33.0, dtype=torch.float64)
    profile = 12.5 * omega.pow(-1.8) * (1.0 + .05 * torch.sin(3 * omega))
    torch_fit = fit_power_law_torch(omega, profile)
    numpy_fit = fit_power_law(omega.numpy(), profile.numpy(), omega_min=.5)

    assert torch_fit["status"] == "ok"
    for key in ("A", "beta", "r_squared"):
        assert torch_fit[key] == pytest.approx(numpy_fit[key], rel=1e-6)

    capped = fit_power_law_torch(
        omega,
        12.5 * omega.pow(-1.8),
        omega_max=8.0,
    )
    assert capped["A"] == pytest.approx(12.5)
    assert capped["beta"] == pytest.approx(1.8)
    assert capped["n_bins"] == 8


def test_direct_profile_sampling_contract():
    freqs = torch.arange(0.0, 9.0)
    profile = 2.0 * freqs
    assert sample_radial_power(freqs, profile, 5.0) == pytest.approx(10.0)
    assert sample_radial_power(freqs, profile, 5.5) == pytest.approx(11.0)
    assert sample_radial_power(freqs, profile, -1.0) == pytest.approx(0.0)
    assert sample_radial_power(freqs, profile, 100.0) == pytest.approx(16.0)
    assert sample_radial_band_power(freqs, profile, 5.0, 1.0) == pytest.approx(10.0)
    assert sample_radial_band_power(freqs, profile, 100.0, 1.0) == pytest.approx(16.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only device contract")
def test_direct_profile_sampling_accepts_cuda_profiles():
    freqs = torch.arange(0.0, 9.0, device="cuda")
    profile = 2.0 * freqs
    assert sample_radial_power(freqs, profile, 5.5) == pytest.approx(11.0)
    assert sample_radial_band_power(freqs, profile, 5.0, 1.0) == pytest.approx(10.0)


def test_log_ema_semantics_and_validation():
    assert update_log_ema(None, 4.0, alpha=.25) == pytest.approx(4.0)
    expected = math.exp(.75 * math.log(4.0) + .25 * math.log(1.0))
    assert update_log_ema(4.0, 1.0, alpha=.25) == pytest.approx(expected)
    assert update_log_ema(9.0, 4.0, alpha=1.0) == pytest.approx(4.0)
    assert update_log_ema(4.0, 0.0, alpha=.25) == pytest.approx(4.0)
    for alpha in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            update_log_ema(4.0, 1.0, alpha=alpha)


def test_failed_and_successful_fits_serialize_as_strict_json():
    failed = build_fit_record(
        fit_power_law_torch(torch.arange(1.0, 17.0), torch.zeros(16))
    )
    assert failed == {
        "status": "fit_failed",
        "A": None,
        "beta": None,
        "r_squared": None,
    }

    omega = torch.arange(1.0, 33.0)
    successful = build_fit_record(
        fit_power_law_torch(omega, 12.5 * omega.pow(-1.8))
    )
    document = {"failed": failed, "successful": successful}
    text = dumps_strict(document)
    assert json.loads(text) == document
    assert "NaN" not in text and "Infinity" not in text
