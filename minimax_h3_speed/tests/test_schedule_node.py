"""Tests for the MiniMaxH3SPEEDSchedule node's plan() method.

Does not require ComfyUI — the Schedule node's plan() is pure Python logic
that delegates to resolve_transition_steps and aligned_speed_sigma.
"""

from __future__ import annotations

import sys
import types
from types import ModuleType

import pytest
import torch

# --- Comfy stubs (must be installed BEFORE importing nodes.*) ---
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

from minimax_h3_speed.config import SCALE_PRESETS
from h3_speed_nodes.helper_nodes.schedule import MiniMaxH3SPEEDSchedule


@pytest.fixture
def sigmas():
    return torch.linspace(1.0, 0.025, 21)


def test_input_types_valid():
    """INPUT_TYPES returns a dict with required keys."""
    types = MiniMaxH3SPEEDSchedule.INPUT_TYPES()
    assert "required" in types
    required = types["required"]
    assert "sigmas" in required
    assert "explicit_preset" in required
    assert "transition_mode" in required
    assert "noise_policy" in required


def test_plan_manual_step_returns_config_and_report(sigmas):
    """manual_step mode returns a SpeedConfig and non-empty report string."""
    node = MiniMaxH3SPEEDSchedule()
    cfg, report = node.plan(
        sigmas=sigmas, explicit_preset="half_then_full",
        transition_mode="manual_step",
    )
    assert cfg is not None
    assert len(cfg.transition_steps) == 1  # half_then_full has one transition
    assert isinstance(report, str)
    assert report  # non-empty


def test_plan_returns_correct_transition_steps(sigmas):
    """manual_sigma mode sets transition_steps from the manual_sigma value."""
    node = MiniMaxH3SPEEDSchedule()
    cfg, _ = node.plan(
        sigmas=sigmas, explicit_preset="half_then_full",
        transition_mode="manual_sigma", manual_sigma=0.5,
    )
    # Find which sigma index first goes below 0.5
    values = [float(s) for s in sigmas]
    expected = next(i for i, v in enumerate(values[:-1]) if v <= 0.5)
    assert cfg.transition_steps[0] == expected


def test_plan_manual_sigma_too_high_raises(sigmas):
    """manual_sigma above all sigmas must raise."""
    node = MiniMaxH3SPEEDSchedule()
    # manual_sigma = 0.024 is below the schedule's last non-zero sigma
    with pytest.raises(ValueError, match="manual sigma is not reached"):
        node.plan(sigmas=sigmas, explicit_preset="half_then_full",
                  transition_mode="manual_sigma", manual_sigma=0.024)


def test_plan_delta_custom_delegates_to_resolve_transition_steps(sigmas):
    """delta_custom mode produces the same steps as resolve_transition_steps."""
    from minimax_h3_speed.h3_runtime import resolve_transition_steps
    from minimax_h3_speed.config import SpeedConfig

    node = MiniMaxH3SPEEDSchedule()
    cfg, report = node.plan(
        sigmas=sigmas, explicit_preset="half_then_full",
        transition_mode="delta_custom", delta=0.01,
        power_A=219.48, power_beta=2.42,
        full_latent_h=45, full_latent_w=80,
    )
    # Cross-check against standalone resolve_transition_steps
    config = SpeedConfig(
        scales=(0.5, 1.0), transition_steps=(5,),
        transition_mode="delta_custom", noise_policy="direct_coarse",
        delta=0.01, power_A=219.48, power_beta=2.42,
        full_latent_h=45, full_latent_w=80,
    )
    expected = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
    assert cfg.transition_steps == expected


def test_plan_report_contains_preset_and_steps(sigmas):
    """Report string contains the preset name and transition steps."""
    node = MiniMaxH3SPEEDSchedule()
    _, report = node.plan(
        sigmas=sigmas, explicit_preset="half_then_full",
        transition_mode="manual_step",
    )
    assert "half_then_full" in report
    assert "scales=" in report
    assert "steps=" in report


def test_all_presets_produce_valid_config(sigmas):
    """Every preset in SCALE_PRESETS yields a non-empty plan."""
    node = MiniMaxH3SPEEDSchedule()
    for preset_name in SCALE_PRESETS:
        cfg, report = node.plan(
            sigmas=sigmas, explicit_preset=preset_name,
            transition_mode="manual_step",
        )
        assert len(cfg.transition_steps) == len(SCALE_PRESETS[preset_name]) - 1, \
            f"preset {preset_name}: steps mismatch"
        assert len(report) > 0


def test_coupled_noise_policy_accepted():
    """noise_policy='coupled_full_grid' is accepted (no crash)."""
    node = MiniMaxH3SPEEDSchedule()
    cfg, _ = node.plan(
        sigmas=torch.linspace(1.0, 0.025, 21),
        explicit_preset="half_then_full",
        transition_mode="manual_step",
        noise_policy="coupled_full_grid",
    )
    assert cfg.noise_policy == "coupled_full_grid"
