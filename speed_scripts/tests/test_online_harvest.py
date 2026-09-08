"""Pure-math tests for the torch-native online spectral analysis (§49)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from conftest import install_comfy_stubs as _install_comfy_stubs


_install_comfy_stubs()

from speed_scripts.harvest import fit_power_law, radial_dct_power
from speed_scripts.online_harvest import (
    SpeedHarvestCollector,
    build_fit_record,
    dumps_strict,
    fit_power_law_torch,
    finite_or_none,
    radial_dct_power_torch,
    sample_radial_band_power,
    sample_radial_power,
    update_log_ema,
)


def _dct_basis_row(k: int, n: int) -> torch.Tensor:
    """The k-th orthonormal DCT-II basis row over n samples (spectral.py convention)."""
    x = torch.arange(n, dtype=torch.float32) + 0.5
    row = torch.cos(math.pi / n * k * x)
    scale = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
    return row * scale


class TestRadialPower:
    def test_known_pure_frequency_lands_in_expected_bin(self):
        """A video equal to one DCT basis function must put all power in that
        function's radial frequency bin."""
        height = width = 16
        freq_k = 3
        basis_2d = torch.einsum(
            "i,j->ij", _dct_basis_row(freq_k, height), _dct_basis_row(0, width)
        )
        video = basis_2d[None, None, None]  # [1, 1, 1, H, W]

        freqs, profile = radial_dct_power_torch(video)

        assert freqs.ndim == 1 and profile.ndim == 1
        assert freqs.numel() == profile.numel()
        assert torch.allclose(freqs, freqs.sort().values)
        assert torch.isfinite(profile).all()
        assert (profile >= 0).all()
        peak = freqs[int(profile.argmax())]
        assert peak == freq_k
        peak_power = float(profile.max())
        rest = float(profile.sum()) - peak_power
        assert peak_power > 100.0 * max(rest, 1e-12)

    def test_constant_video_puts_all_power_at_dc(self):
        video = torch.full((1, 1, 1, 8, 8), 0.25)
        freqs, profile = radial_dct_power_torch(video)
        assert int(freqs[0]) == 0
        assert float(profile[0]) > 0
        assert float(profile[1:].sum()) < 1e-6 * float(profile[0])

    def test_rejects_non_video_input(self):
        with pytest.raises(ValueError):
            radial_dct_power_torch(torch.zeros(4, 16, 16))


class TestTorchVsNumpy:
    def test_random_video_matches_numpy_path(self):
        torch.manual_seed(7)
        video = torch.randn(2, 3, 2, 16, 24)

        freqs_np, profile_np = radial_dct_power(video)
        freqs_t, profile_t = radial_dct_power_torch(video)

        common = np.intersect1d(freqs_np, freqs_t.cpu().numpy())
        assert common.size >= 8
        np_at = profile_np[np.searchsorted(freqs_np, common)]
        t_at = profile_t.cpu().numpy()[np.searchsorted(freqs_t.cpu().numpy(), common)]
        assert np.allclose(np_at, t_at, rtol=1e-4, atol=1e-7)

    def test_second_call_same_resolution_uses_cache(self):
        torch.manual_seed(11)
        video = torch.randn(1, 1, 1, 8, 8)
        freqs_a, profile_a = radial_dct_power_torch(video)
        freqs_b, profile_b = radial_dct_power_torch(video * 2.0)
        assert torch.equal(freqs_a, freqs_b)
        assert torch.allclose(profile_b, profile_a * 4.0)


class TestFitPowerLaw:
    def test_known_power_law_recovered(self):
        omega = torch.arange(1.0, 33.0)
        profile = 12.5 * omega.pow(-1.8)
        fit = fit_power_law_torch(omega, profile)
        assert fit["status"] == "ok"
        assert fit["A"] == pytest.approx(12.5)
        assert fit["beta"] == pytest.approx(1.8)
        assert fit["r_squared"] > 0.99999
        assert fit["n_bins"] == 32
        assert fit["health"] == "good"

    def test_omega_max_caps_fit_range(self):
        omega = torch.arange(1.0, 33.0)
        profile = 12.5 * omega.pow(-1.8)
        fit = fit_power_law_torch(omega, profile, omega_max=8.0)
        assert fit["status"] == "ok"
        assert fit["n_bins"] == 8
        assert fit["A"] == pytest.approx(12.5)
        assert fit["beta"] == pytest.approx(1.8)

    def test_torch_fitter_agrees_with_numpy_fitter(self):
        omega = torch.arange(1.0, 33.0, dtype=torch.float64)
        jitter = 1.0 + 0.05 * torch.sin(3.0 * omega)
        profile = 12.5 * omega.pow(-1.8) * jitter

        torch_fit = fit_power_law_torch(omega, profile)
        numpy_fit = fit_power_law(
            omega.numpy(), profile.numpy(), omega_min=0.5
        )

        assert torch_fit["status"] == "ok"
        for key in ("A", "beta", "r_squared"):
            assert math.isclose(torch_fit[key], numpy_fit[key], rel_tol=1e-6)
        assert torch_fit["n_bins"] == numpy_fit["n_bins"]


class TestDirectSampling:
    def test_point_sample_exact_bin_and_midpoint(self):
        freqs = torch.arange(0.0, 9.0)
        profile = 2.0 * freqs  # linear: P(x) = 2x
        assert sample_radial_power(freqs, profile, 5.0) == 10.0
        assert sample_radial_power(freqs, profile, 5.5) == 11.0

    def test_point_sample_clamps_outside_range(self):
        freqs = torch.arange(0.0, 9.0)
        profile = 2.0 * freqs
        assert sample_radial_power(freqs, profile, -1.0) == 0.0
        assert sample_radial_power(freqs, profile, 100.0) == 16.0

    def test_band_mean_averages_bins_in_band(self):
        freqs = torch.arange(0.0, 9.0)
        profile = 2.0 * freqs
        band = sample_radial_band_power(freqs, profile, 5.0, half_width=1.0)
        assert band == (8.0 + 10.0 + 12.0) / 3.0

    def test_empty_band_falls_back_to_point_sample(self):
        freqs = torch.arange(0.0, 9.0)
        profile = 2.0 * freqs
        band = sample_radial_band_power(freqs, profile, 100.0, half_width=1.0)
        assert band == sample_radial_power(freqs, profile, 100.0)

    def test_unsorted_profile_input_is_sorted_before_sampling(self):
        freqs = torch.tensor([4.0, 2.0, 6.0])
        profile = torch.tensor([8.0, 4.0, 12.0])  # P(x) = 2x, scrambled order
        assert sample_radial_power(freqs, profile, 5.0) == 10.0

    def test_point_sample_builds_operand_on_input_device(self, monkeypatch):
        """The searchsorted operand is built from the input tensor
        (``x.new_tensor``), never with an implicit CPU ``torch.tensor``."""
        import speed_scripts.online_harvest as online_harvest

        def no_implicit_cpu_tensor(*args, **kwargs):
            raise AssertionError("implicit torch.tensor() call in sampling path")

        monkeypatch.setattr(online_harvest.torch, "tensor", no_implicit_cpu_tensor)
        freqs = torch.arange(0.0, 9.0)
        profile = 2.0 * freqs
        assert sample_radial_power(freqs, profile, 5.0) == 10.0
        assert sample_radial_power(freqs, profile, 5.5) == 11.0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="device-mismatch guard needs CUDA; runs on GPU hosts",
    )
    def test_point_sample_cuda_profile_stays_on_device(self):
        """A CUDA profile must not crash searchsorted with a CPU operand."""
        freqs = torch.arange(0.0, 9.0, device="cuda")
        profile = 2.0 * freqs
        assert sample_radial_power(freqs, profile, 5.5) == pytest.approx(11.0)
        assert sample_radial_band_power(freqs, profile, 5.0, half_width=1.0) == pytest.approx(
            (8.0 + 10.0 + 12.0) / 3.0
        )


class TestEmaAndJsonSafety:
    def test_ema_seeds_from_first_valid_measurement(self):
        assert update_log_ema(None, 4.0) == 4.0

    def test_ema_log_space_update(self):
        ema = update_log_ema(4.0, 1.0, alpha=0.25)
        expected = math.exp(0.75 * math.log(4.0) + 0.25 * math.log(1.0))
        assert math.isclose(ema, expected)

    def test_ema_alpha_one_equals_raw_measurement(self):
        """alpha=1.0 disables smoothing: the EMA is the raw measurement,
        both when seeding and when a previous EMA exists."""
        assert update_log_ema(None, 4.0, alpha=1.0) == 4.0
        assert update_log_ema(9.0, 4.0, alpha=1.0) == 4.0

    def test_ema_ignores_nonpositive_raw(self):
        assert update_log_ema(4.0, 0.0) == 4.0
        assert update_log_ema(4.0, -1.0) == 4.0

    def test_ema_rejects_alpha_outside_open_closed_interval(self):
        for alpha in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="alpha"):
                update_log_ema(4.0, 1.0, alpha=alpha)

    def test_collector_accepts_full_alpha_range_and_rejects_outside(self):
        """The collector boundary validates smoothing_alpha over (0, 1]."""

        def collector(smoothing_alpha):
            return SpeedHarvestCollector(
                delta=0.01,
                noise_amplitude=1.0,
                noise_decay_exponent=1.0,
                smoothing_alpha=smoothing_alpha,
            )

        assert collector(1.0).smoothing_alpha == 1.0
        for alpha in (0.0, -0.5, 1.5):
            with pytest.raises(ValueError, match="smoothing_alpha"):
                collector(alpha)

    def test_finite_or_none(self):
        assert finite_or_none(1.5) == 1.5
        assert finite_or_none(float("nan")) is None
        assert finite_or_none(float("inf")) is None
        assert finite_or_none(None) is None

    def test_zero_profile_gives_failed_record_without_nan(self):
        freqs = torch.arange(1.0, 17.0)
        profile = torch.zeros(16)
        fit = fit_power_law_torch(freqs, profile)
        assert fit["status"] == "fit_failed"

        record = build_fit_record(fit)
        assert record == {
            "status": "fit_failed",
            "A": None,
            "beta": None,
            "r_squared": None,
        }
        json.dumps(record, allow_nan=False)  # must not raise

    def test_nan_profile_gives_failed_record(self):
        freqs = torch.arange(1.0, 17.0)
        profile = torch.full((16,), float("nan"))
        fit = fit_power_law_torch(freqs, profile)
        assert fit["status"] == "fit_failed"
        record = build_fit_record(fit)
        json.dumps(record, allow_nan=False)

    def test_successful_fit_record_is_strict_json_safe(self):
        omega = torch.arange(1.0, 33.0)
        profile = 12.5 * omega.pow(-1.8)
        record = build_fit_record(fit_power_law_torch(omega, profile))
        assert record["status"] == "ok"
        assert record["A"] == pytest.approx(12.5)
        document = {"x0_signal": {"fit": record}}
        assert json.loads(dumps_strict(document)) == document
