"""Contract tests for the manual explicit step-through sampler node."""
import importlib

import pytest
import torch

from speed_scripts.config import SpeedConfig

# Canonical comfy stubs + shared fakes live in conftest now.
from conftest import (
    install_comfy_stubs as _install_comfy_stubs,
    make_fake_guider,
    make_fake_noise,
    make_nested,
)

_install_comfy_stubs()


def _fake_run_env():
    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_nested(video, audio)
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.0, 20)
    calls = []
    return make_fake_noise(), make_fake_guider(calls), sigmas, latent, calls


def test_node_exports_manual_sampler():
    mod = importlib.import_module("sampler_node_manual")
    cls = mod.MiniMaxH3SPEEDSamplerManual
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert cls.FUNCTION == "sample"
    assert "MiniMaxH3SPEEDSamplerManual" in mod.NODE_CLASS_MAPPINGS


def test_input_schema_manual_widgets():
    mod = importlib.import_module("sampler_node_manual")
    required = mod.MiniMaxH3SPEEDSamplerManual.INPUT_TYPES()["required"]
    for key in ("noise", "guider", "sigmas", "latent_image",
                "ratio_mode", "transition_goal_1", "transition_resolution_1"):
        assert key in required, f"missing required input: {key}"
    # defaults must form a valid quarter_half_3q_full-style schedule
    assert required["transition_resolution_1"][1]["default"] == 0.25
    assert required["transition_resolution_4"][1]["default"] == 1.0


def test_manual_sample_runs_all_default_stages():
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, calls = _fake_run_env()
    out, denoised = mod.MiniMaxH3SPEEDSamplerManual().sample(
        noise, guider, sigmas, latent
    )
    assert out is not None and out.get("samples") is not None
    # defaults: 4 active stages (0.25 / 0.5 / 0.75 / 1.0)
    assert len(calls) == 4, f"expected 4 stages, got {len(calls)}"


def test_manual_goal_zero_skips_stages():
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, calls = _fake_run_env()
    node = mod.MiniMaxH3SPEEDSamplerManual()
    # disable stage 3 (goal 0) -> quarter → half → full: 3 stages
    out, _ = node.sample(noise, guider, sigmas, latent, transition_goal_3=0)
    assert len(calls) == 3, f"expected 3 stages, got {len(calls)}"
    assert out is not None


def test_manual_single_stage_raises():
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, _ = _fake_run_env()
    node = mod.MiniMaxH3SPEEDSamplerManual()
    with pytest.raises(ValueError, match="at least two active stages"):
        node.sample(noise, guider, sigmas, latent,
                    transition_goal_2=0, transition_goal_3=0, transition_goal_4=0)


# ---------------------------------------------------------------------------
# Regression: ratio-mode scale calculation
#
# The stage scale is `resolution` in BOTH modes; `goal` only positions the
# boundary. The old behavior multiplied the scale by the goal
# (scale = resolution * goal), silently producing wrong ladders like
# (goal 0.6, resolution 0.5) -> scale 0.3.
# ---------------------------------------------------------------------------


def test_ratio_mode_scale_is_resolution_not_resolution_times_goal(monkeypatch):
    """(goal 0.6, resolution 0.5) must build scales (0.5, 1.0), not (0.3, 1.0).

    Captures the SpeedConfig the node hands to run_speed_pipeline: the old
    bug multiplied the scale by the goal (0.5 * 0.6 = 0.3), which still
    passed SpeedConfig's strictly-increasing check and silently ran the
    wrong ladder.
    """
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, _ = _fake_run_env()

    captured = {}

    def _spy_run(r_noise, r_guider, r_sigmas, r_latent, config, **kwargs):
        captured["config"] = config
        return r_latent, r_latent

    monkeypatch.setattr(mod, "run_speed_pipeline", _spy_run)
    node = mod.MiniMaxH3SPEEDSamplerManual()
    out, _ = node.sample(noise, guider, sigmas, latent,
                         ratio_mode="ratio",
                         transition_goal_1=0.6, transition_resolution_1=0.5,
                         transition_goal_2=0.8, transition_resolution_2=1.0,
                         transition_goal_3=0, transition_resolution_3=0,
                         transition_goal_4=0, transition_resolution_4=0)

    assert out is not None
    config = captured["config"]
    assert config.scales == (0.5, 1.0), (
        f"ratio mode must keep resolution as the stage scale, got {config.scales}"
    )
    # Boundaries: round(0.6 * 19) = 11 for stage 0; final stage to the end.
    assert config.transition_steps == (11,)


def test_ratio_mode_boundary_position_rounds_goal_times_total_steps():
    """Boundary = round(goal * total_steps); goals (0.3, 0.6) on 19 steps -> (6, 11)."""
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, calls = _fake_run_env()  # 20 sigmas = 19 steps
    node = mod.MiniMaxH3SPEEDSamplerManual()
    node.sample(noise, guider, sigmas, latent,
                ratio_mode="ratio",
                transition_goal_1=0.3, transition_resolution_1=0.25,
                transition_goal_2=0.6, transition_resolution_2=0.5,
                transition_goal_3=0, transition_resolution_3=0,
                transition_goal_4=1.0, transition_resolution_4=1.0)
    # goals 3 and 4 both zero/unused: stage 3 skipped (goal 0), stage 4 is
    # the final stage (its goal is unused — runs to the end).
    assert len(calls) == 3
    # Stage slices: [0..6], [6..11], [11..19] (inclusive boundary indices).
    assert calls[0] == 7, f"stage 0 must cover steps 0..6 (7 sigmas), got {calls[0]}"
    assert calls[1] == 6, f"stage 1 must cover steps 6..11 (6 sigmas), got {calls[1]}"
    assert calls[2] == 9, f"final stage must cover steps 11..19 (9 sigmas), got {calls[2]}"


def test_ratio_mode_goal_above_one_rejected():
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, _ = _fake_run_env()
    node = mod.MiniMaxH3SPEEDSamplerManual()
    with pytest.raises(ValueError, match="fraction of the schedule"):
        node.sample(noise, guider, sigmas, latent,
                    ratio_mode="ratio",
                    transition_goal_1=7.5, transition_resolution_1=0.25)


def test_steps_mode_fractional_goal_rejected_not_truncated():
    """int(5.5) silently became 5; it must raise instead."""
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, _ = _fake_run_env()
    node = mod.MiniMaxH3SPEEDSamplerManual()
    with pytest.raises(ValueError, match="whole step index"):
        node.sample(noise, guider, sigmas, latent,
                    ratio_mode="steps",
                    transition_goal_1=3, transition_resolution_1=0.25,
                    transition_goal_2=5.5, transition_resolution_2=0.5)


def test_unsupported_ratio_mode_rejected():
    mod = importlib.import_module("sampler_node_manual")
    noise, guider, sigmas, latent, _ = _fake_run_env()
    node = mod.MiniMaxH3SPEEDSamplerManual()
    with pytest.raises(ValueError, match="unsupported ratio_mode"):
        node.sample(noise, guider, sigmas, latent, ratio_mode="fractions")
