"""Contract tests for the manual (DEPRECATED - brittle, prefer delta_custom) step-through sampler node."""
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
