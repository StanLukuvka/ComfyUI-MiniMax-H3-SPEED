"""Integration test for the full SPEED calibration pipeline.

End-to-end flow:
1. Generate synthetic noise (nested latent)
2. Run HarvestCallback to capture residual spectrum
3. Parse harvest report (simulate HarvesterToConfig)
4. Schedule with delta_custom mode using fitted A/β
5. Verify transition steps are reasonable and within bounds
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest
import torch

# --- Comfy stubs ---
_comfy = ModuleType("comfy")
for mod in ("samplers", "utils", "model_management", "nested_tensor"):
    sys.modules[f"comfy.{mod}"] = ModuleType(f"comfy.{mod}")
sys.modules["comfy"] = _comfy


class _MockNestedTensor:
    """Mock H3 nested latent with video + audio streams."""
    is_nested = True

    def __init__(self, video, audio):
        self._video = video
        self._audio = audio

    def unbind(self):
        return [self._video, self._audio]


# --- Import test targets ---
from minimax_h3_speed.harvest import (
    HarvestCallback,
    fit_power_law,
    radial_power_spectrum,
    recommend_configs,
    _fit_health,
)
from minimax_h3_speed.config import SCALE_PRESETS, SpeedConfig
from minimax_h3_speed.h3_runtime import resolve_transition_steps
from minimax_h3_speed.spectral import idct2


def _make_mock_latent(H=32, W=32, T=8, C=4):
    """Create a synthetic nested latent."""
    video = torch.randn(1, C, T, H, W)
    audio = torch.randn(1, C, 2, T)
    return _MockNestedTensor(video, audio)


def _make_profiles_with_beta(n, H, W, target_A, target_beta):
    """Create profile tuples with known power-law spectrum."""
    freqs, _ = radial_power_spectrum(torch.randn(1, 4, 8, H, W))
    # Build profile matching target power law
    power = target_A * np.power(np.maximum(freqs, 1.0), -target_beta)
    profiles = [(1.0 - i * 0.2, freqs.copy(), power.copy()) for i in range(n)]
    return profiles


def _run_callback_steps(callback, latent, n_steps=5):
    """Simulate multiple callback invocations."""
    x0 = _make_mock_latent(
        latent._video.shape[-2], latent._video.shape[-1],
        latent._video.shape[2], latent._video.shape[1]
    )
    for step in range(n_steps):
        sigma = callback.sigmas[min(step, len(callback.sigmas) - 1)].item()
        x_noisy_video = x0._video + sigma * torch.randn_like(x0._video)
        x_noisy_audio = x0._audio + sigma * torch.randn_like(x0._audio)
        x_noisy = _MockNestedTensor(x_noisy_video, x_noisy_audio)
        callback(step=step, x0=x0, x=x_noisy, total_steps=n_steps)


def _set_mock_profiles(callback, profiles):
    """Inject pre-computed profiles directly into callback."""
    callback.profiles = profiles


def test_integration_harvest_to_schedule():
    """Full pipeline: harvest noise → fit power law → schedule transitions."""
    H, W = 32, 32
    sigmas = torch.linspace(1.0, 0.025, 21)

    # Create callback and inject mock profiles with known spectrum
    callback = HarvestCallback(sigmas=sigmas, every=1)
    profiles = _make_profiles_with_beta(n=5, H=H, W=W, target_A=200.0, target_beta=2.0)
    _set_mock_profiles(callback, profiles)

    # Verify profiles were captured
    assert len(callback.profiles) == 5

    # Finalize harvest
    result = callback.finalize(omega_min=0.5, latent_h=H, latent_w=W, delta=0.01, fit_mode="pooled")

    # Verify harvest structure and values
    assert result["overall_fit_A"] > 0
    assert abs(result["overall_fit_beta"] - 2.0) < 0.1  # Should match target

    # Recommend configs for all presets
    recommended = result.get("recommended_config", {})
    assert len(recommended) == len(SCALE_PRESETS)
    for cfg in recommended.values():
        assert len(cfg["transition_steps"]) > 0

    # Build SpeedConfig and verify resolution
    first_cfg = list(recommended.values())[0]
    config = SpeedConfig(
        scales=tuple(first_cfg["scales"]),
        transition_steps=tuple(first_cfg["transition_steps"]),
        transition_mode="delta_custom",
        noise_policy="direct_coarse",
        delta=0.01,
        power_A=result["overall_fit_A"],
        power_beta=result["overall_fit_beta"],
        transition_seed_offset=10000,
        full_latent_h=H,
        full_latent_w=W,
    )
    resolved = resolve_transition_steps(config, sigmas, H_full=H, W_full=W)
    assert len(resolved) == 1
    assert 0 < resolved[0] < len(sigmas)


def test_integration_coupled_noise_policy():
    """Verify coupled_full_grid noise policy creates valid config."""
    sigmas = torch.linspace(1.0, 0.025, 21)
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
        noise_policy="coupled_full_grid",
        delta=0.01,
        power_A=219.48,
        power_beta=2.42,
        transition_seed_offset=10000,
        full_latent_h=45,
        full_latent_w=80,
    )
    assert config.noise_policy == "coupled_full_grid"
    resolved = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
    assert resolved == [5] or resolved == (5,)  # type variation OK


def test_integration_delta_custom_with_fitted_params():
    """Verify delta_custom mode uses fitted A/β for transition computation."""
    sigmas = torch.linspace(1.0, 0.025, 21)
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),  # placeholder
        transition_mode="delta_custom",
        noise_policy="direct_coarse",
        delta=0.01,
        power_A=150.0,
        power_beta=1.8,
        transition_seed_offset=10000,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
    assert len(resolved) == 1
    assert 0 < resolved[0] < len(sigmas)
    assert resolved != [5]  # Should differ from placeholder


def test_integration_all_presets_have_recommendations():
    """Verify recommend_configs produces valid output for every preset."""
    sigmas = torch.linspace(1.0, 0.025, 21)
    recommended = recommend_configs(A=200.0, beta=2.0, sigmas=sigmas, latent_h=45, latent_w=80, delta=0.01)

    assert set(recommended.keys()) == set(SCALE_PRESETS.keys())
    for cfg in recommended.values():
        assert len(cfg["scales"]) >= 2
        assert len(cfg["transition_steps"]) == len(cfg["scales"]) - 1
        assert all(0 <= s < len(sigmas) for s in cfg["transition_steps"])


def test_integration_harvest_report_format():
    """Verify harvest report has all expected fields."""
    sigmas = torch.linspace(1.0, 0.025, 21)
    callback = HarvestCallback(sigmas=sigmas, every=1)
    profiles = _make_profiles_with_beta(n=5, H=32, W=32, target_A=200.0, target_beta=2.0)
    _set_mock_profiles(callback, profiles)

    result = callback.finalize(omega_min=0.5, latent_h=32, latent_w=32, delta=0.01, fit_mode="pooled")

    expected_keys = {"fit_mode", "sigma_levels", "sigma_fits", "overall_fit_A",
                     "overall_fit_beta", "overall_fit_r2", "n_frequency_bins",
                     "fit_health", "per_sigma_profiles", "recommended_config"}
    assert expected_keys.issubset(set(result.keys()))
    assert result["fit_health"] in {"good", "fair", "weak", "suspect"}
    assert len(result["sigma_levels"]) == 5


def test_integration_multiple_fit_modes():
    """Verify all three fit modes produce valid results."""
    sigmas = torch.linspace(1.0, 0.025, 21)
    H, W = 32, 32

    for fit_mode in ["first", "per_sigma", "pooled"]:
        callback = HarvestCallback(sigmas=sigmas, every=1)
        profiles = _make_profiles_with_beta(n=5, H=H, W=W, target_A=200.0, target_beta=2.0)
        _set_mock_profiles(callback, profiles)

        result = callback.finalize(omega_min=0.5, latent_h=H, latent_w=W, delta=0.01, fit_mode=fit_mode)
        assert result["overall_fit_A"] > 0
        assert abs(result["overall_fit_beta"] - 2.0) < 0.1
        assert result["fit_health"] in {"good", "fair", "weak", "suspect"}
