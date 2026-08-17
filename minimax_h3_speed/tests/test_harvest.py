"""Tests for the sigma-harvest module."""

from __future__ import annotations

import math
import sys
from types import ModuleType

import numpy as np
import pytest
import torch


# --- Comfy stubs (must be installed BEFORE importing minimax_h3_speed) ---
_comfy = ModuleType("comfy")
_comfy.samplers = ModuleType("comfy.samplers")
_comfy.utils = ModuleType("comfy.utils")
_comfy.model_management = ModuleType("comfy.model_management")
_comfy.nested_tensor = ModuleType("comfy.nested_tensor")
sys.modules["comfy"] = _comfy
sys.modules["comfy.samplers"] = _comfy.samplers
sys.modules["comfy.utils"] = _comfy.utils
sys.modules["comfy.model_management"] = _comfy.model_management
sys.modules["comfy.nested_tensor"] = _comfy.nested_tensor


class _MockNestedTensor:
    """Mock H3 nested latent with video + audio streams."""
    is_nested = True

    def __init__(self, video, audio):
        self._streams = [video, audio]

    def unbind(self):
        return self._streams


# --- Mock helpers (defined before imports so they're available in tests) ---
def _make_callback(n_sigmas=5, every=1):
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.2, 0.05, 0.0])
    from minimax_h3_speed.harvest import HarvestCallback
    return HarvestCallback(sigmas=sigmas, every=every)


def _make_mock_latent(H=32, W=32, T=8, C=4):
    video = torch.randn(1, C, T, H, W)
    audio = torch.randn(1, C, 2, T)
    return _MockNestedTensor(video, audio)


# --- Import test targets (after stubs installed) ---
from minimax_h3_speed.harvest import (
    HarvestCallback,
    _fit_health,
    fit_power_law,
    radial_power_spectrum,
    recommend_configs,
)
from minimax_h3_speed.config import SCALE_PRESETS, SpeedConfig


class TestRadialPowerSpectrum:
    def test_returns_freqs_and_profile(self):
        video = torch.randn(1, 4, 8, 32, 32)
        freqs, profile = radial_power_spectrum(video)
        assert isinstance(freqs, np.ndarray)
        assert isinstance(profile, np.ndarray)
        assert len(freqs) == len(profile)
        assert len(freqs) > 0

    def test_dc_bin_has_power(self):
        video = torch.ones(1, 1, 4, 64, 64)
        freqs, profile = radial_power_spectrum(video)
        assert freqs[0] == 0
        assert profile[0] > 0

    def test_frequency_bins_are_non_negative(self):
        video = torch.randn(1, 2, 4, 16, 32)
        freqs, _ = radial_power_spectrum(video)
        assert np.all(freqs >= 0)
        assert np.all(np.diff(freqs) >= 0)


class TestFitPowerLaw:
    def test_recovers_known_params(self):
        A_true, beta_true = 100.0, 2.0
        freqs = np.logspace(-0.5, 2.5, 50)
        profile = A_true * freqs ** (-beta_true) + 0.01 * np.random.randn(50)
        fit = fit_power_law(freqs, profile, omega_min=0.5)
        assert fit["A"] > 0
        assert fit["beta"] > 0
        assert abs(fit["beta"] - beta_true) < 0.5

    def test_insufficient_bins_raises(self):
        freqs = np.array([1.0, 2.0])
        profile = np.array([1.0, 0.5])
        with pytest.raises(ValueError, match="not enough frequency bins"):
            fit_power_law(freqs, profile)

    def test_r_squared_in_range(self):
        freqs = np.logspace(0, 2, 30)
        profile = 50.0 * freqs ** (-1.5)
        fit = fit_power_law(freqs, profile)
        assert 0 <= fit["r_squared"] <= 1


class TestFitHealth:
    def test_good_fit(self):
        fit = {"A": 100, "beta": 2.0, "r_squared": 0.9}
        assert _fit_health(fit) == "good"

    def test_fair_fit(self):
        fit = {"A": 100, "beta": 1.5, "r_squared": 0.5}
        assert _fit_health(fit) == "fair"

    def test_weak_fit(self):
        fit = {"A": 100, "beta": 0.5, "r_squared": 0.3}
        assert _fit_health(fit) == "weak"

    def test_suspect_negative_beta(self):
        fit = {"A": 100, "beta": -1.0, "r_squared": 0.8}
        assert _fit_health(fit) == "suspect"

    def test_invalid_nan(self):
        fit = {"A": float("nan"), "beta": 2.0, "r_squared": 0.9}
        assert _fit_health(fit) == "invalid"

        fit = {"A": 100.0, "beta": float("nan"), "r_squared": 0.9}
        assert _fit_health(fit) == "invalid"

        fit = {"A": 100.0, "beta": 2.0, "r_squared": float("nan")}
        assert _fit_health(fit) == "invalid"


class TestHarvestCallback:
    def test_captures_residual(self):
        cb = _make_callback()
        video = torch.randn(1, 4, 8, 32, 32)
        audio = torch.randn(1, 4, 2, 8)
        x = _MockNestedTensor(video, audio)
        x0 = _MockNestedTensor(video.clone() * 0.9, audio.clone() * 0.9)
        cb(0, x0, x, total_steps=5)
        assert len(cb.profiles) == 1
        sigma, freqs, profile = cb.profiles[0]
        assert abs(sigma - 1.0) < 1e-6
        assert len(freqs) > 0

    def test_skips_intermediate_steps(self):
        cb = _make_callback(every=2)
        video = torch.randn(1, 4, 8, 32, 32)
        audio = torch.randn(1, 4, 2, 8)
        x = _MockNestedTensor(video, audio)
        x0 = _MockNestedTensor(video.clone() * 0.5, audio.clone() * 0.5)
        cb(0, x0, x, total_steps=5)
        cb(1, x0, x, total_steps=5)
        cb(2, x0, x, total_steps=5)
        cb(4, x0, x, total_steps=5)
        assert len(cb.profiles) == 3

    def test_rejects_bad_shapes(self):
        cb = _make_callback()
        x = torch.randn(1, 4, 32, 32)
        x0 = torch.zeros_like(x)
        cb(0, x0, x, total_steps=5)
        assert len(cb.profiles) == 0

    def test_rejects_non_nested(self):
        cb = _make_callback()

        class BadTensor:
            is_nested = False

            def unbind(self):
                return [torch.randn(1, 4, 8, 32, 32)]

        x = BadTensor()
        x0 = BadTensor()
        cb(0, x0, x, total_steps=5)
        assert len(cb.profiles) == 0

    def test_finalize_requires_profiles(self):
        cb = _make_callback()
        with pytest.raises(RuntimeError, match="no latents captured"):
            cb.finalize(omega_min=0.5, latent_h=32, latent_w=32, delta=0.01,
                        fit_mode="first")


class TestFinalize:
    def _build_full_callback(self, n_levels=3):
        from minimax_h3_speed.harvest import HarvestCallback
        sigmas = torch.tensor([1.0, 0.6, 0.3, 0.05])
        cb = HarvestCallback(sigmas=sigmas, every=1)
        video = torch.randn(1, 4, 8, 32, 32)
        audio = torch.randn(1, 4, 2, 8)
        for i in range(n_levels):
            x = _MockNestedTensor(video.clone(), audio.clone())
            x0 = _MockNestedTensor(
                video.clone() * (1 - float(sigmas[i])),
                audio.clone() * (1 - float(sigmas[i]))
            )
            cb(i, x0, x, total_steps=n_levels)
        return cb

    def test_finalize_first_mode(self):
        cb = self._build_full_callback()
        result = cb.finalize(omega_min=0.5, latent_h=32, latent_w=32,
                             delta=0.01, fit_mode="first")
        assert "overall_fit_A" in result
        assert "overall_fit_beta" in result
        assert result["fit_mode"] == "first"
        assert result["fit_health"] in ("good", "fair", "weak", "suspect", "invalid")

    def test_finalize_per_sigma_mode(self):
        cb = self._build_full_callback()
        result = cb.finalize(fit_mode="per_sigma")
        assert "sigma_fits" in result
        assert len(result["sigma_fits"]) > 0

    def test_finalize_pooled_mode(self):
        cb = self._build_full_callback()
        result = cb.finalize(fit_mode="pooled")
        assert "overall_fit_A" in result

    def test_finalize_unknown_mode_raises(self):
        cb = self._build_full_callback()
        with pytest.raises(ValueError, match="unknown fit_mode"):
            cb.finalize(fit_mode="banana")

    def test_recommend_configs_produces_steps(self):
        cb = self._build_full_callback()
        result = cb.finalize(latent_h=32, latent_w=32, delta=0.01,
                             fit_mode="first")
        rec = result.get("recommended_config")
        assert rec is not None
        for name, preset_data in rec.items():
            assert "scales" in preset_data
            assert "transition_steps" in preset_data
            assert len(preset_data["scales"]) == len(preset_data["transition_steps"]) + 1


class TestRecommendConfigs:
    def test_steps_count_matches_transitions(self):
        from minimax_h3_speed.harvest import recommend_configs
        sigmas = torch.tensor([1.0, 0.7, 0.4, 0.2, 0.05, 0.0])
        rec = recommend_configs(100.0, 2.0, sigmas, latent_h=32, latent_w=32)
        for name, data in rec.items():
            n_scales = len(data["scales"])
            n_steps = len(data["transition_steps"])
            assert n_steps == n_scales - 1, f"{name}: {n_steps} steps for {n_scales} scales"

    def test_all_presets_present(self):
        sigmas = torch.tensor([1.0, 0.5, 0.0])
        rec = recommend_configs(100.0, 2.0, sigmas, latent_h=16, latent_w=32)
        for name in SCALE_PRESETS:
            assert name in rec

    def test_steps_are_valid_indices(self):
        n_sigmas = 20
        sigmas = torch.linspace(1.0, 0.0, n_sigmas + 1)
        rec = recommend_configs(100.0, 2.0, sigmas, latent_h=45, latent_w=80)
        for name, data in rec.items():
            for step in data["transition_steps"]:
                assert 0 <= step < n_sigmas, f"step {step} out of range"

    def test_recommended_steps_monotonic_non_decreasing(self):
        """For monotonically decreasing power spectrum (Eq. 4), earlier scales
        (lower resolution) should activate later in denoising (higher step index).
        """
        sigmas = torch.linspace(1.0, 0.0, 51)  # 50 steps
        rec = recommend_configs(219.48, 2.42, sigmas, latent_h=45, latent_w=80)
        for name, data in rec.items():
            steps = data["transition_steps"]
            if len(steps) >= 2:
                assert steps[0] <= steps[1], \
                    f"{name}: steps {steps} not non-decreasing"

    def test_resolve_transition_steps_delta_custom_matches_recommend(self):
        """Given config with transition_mode='delta_custom', the resolved steps
        must equal recommend_configs output for the same parameters."""
        from minimax_h3_speed.h3_runtime import resolve_transition_steps
        sigmas = torch.linspace(1.0, 0.0, 21)  # 20 steps
        config = SpeedConfig(
            scales=(0.5, 1.0),
            transition_steps=(5,),
            transition_mode="delta_custom",
            delta=0.01,
            power_A=219.48,
            power_beta=2.42,
            full_latent_h=45,
            full_latent_w=80,
        )
        resolved = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
        rec = recommend_configs(219.48, 2.42, sigmas, latent_h=45, latent_w=80)
        expected = rec["half_then_full"]["transition_steps"]
        assert resolved == tuple(expected)
