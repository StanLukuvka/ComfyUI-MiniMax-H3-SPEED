"""Automatic node/config public contracts."""

import importlib

import pytest

from conftest import make_latent
from speed_scripts.automatic_config import (
    PRESET_TO_STAGES,
    STAGES_TO_SCALES,
    build_automatic_speed_config,
)


def _capture_node_config(monkeypatch, **overrides):
    mod = importlib.import_module("sampler_node")
    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kwargs):
        captured["config"] = config
        return latent_image, latent_image

    monkeypatch.setattr(mod, "run_speed_pipeline", fake_pipeline)
    mod.MiniMaxH3SPEEDSampler().sample(
        noise=object(),
        guider=type("Guider", (), {})(),
        sigmas=None,
        latent_image=make_latent(h=45, w=80),
        **overrides,
    )
    return captured["config"]


def test_automatic_node_public_surface_and_defaults():
    cls = importlib.import_module("sampler_node").MiniMaxH3SPEEDSampler
    required = cls.INPUT_TYPES()["required"]
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert {"noise", "guider", "sigmas", "latent_image", "stages",
            "noise_policy", "Tolerance (Delta)", "noise_amplitude",
            "noise_decay_exponent", "seed_offset"} <= set(required)
    assert "preset" not in required and "transition_mode" not in required
    assert required["stages"] == ("INT", {"default": 3, "min": 2, "max": 4})
    assert required["Tolerance (Delta)"][1]["default"] == 0.005
    assert required["noise_amplitude"][1]["default"] == 12.105
    assert required["noise_decay_exponent"][1]["default"] == 0.773
    assert required["noise_amplitude"][1]["step"] == 0.0001
    assert required["noise_amplitude"][1]["round"] == 0.0001
    assert required["noise_decay_exponent"][1]["step"] == 0.0001
    assert required["noise_decay_exponent"][1]["round"] == 0.0001
    assert required["Tolerance (Delta)"][1]["step"] == 0.001


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_builder_maps_stage_count_to_exact_scale_ladder(stages):
    cfg = build_automatic_speed_config(
        make_latent(h=24, w=40),
        stages=stages,
        noise_policy="direct_coarse",
        delta=0.005,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
    )
    assert cfg.scales == STAGES_TO_SCALES[stages]
    assert cfg.transition_steps == tuple(range(1, stages))
    assert cfg.transition_mode == "delta_custom"
    assert (cfg.full_latent_h, cfg.full_latent_w) == (24, 40)


def test_node_forwards_generation_configuration_to_shared_builder(monkeypatch):
    cfg = _capture_node_config(
        monkeypatch,
        stages=3,
        noise_policy="coupled_full_grid",
        delta=0.007,
        noise_amplitude=13.5,
        noise_decay_exponent=0.9,
        seed_offset=777,
    )
    assert cfg.scales == STAGES_TO_SCALES[3]
    assert cfg.transition_mode == "delta_custom"
    assert cfg.noise_policy == "coupled_full_grid"
    assert (cfg.delta, cfg.noise_amplitude, cfg.noise_decay_exponent) == (0.007, 13.5, 0.9)
    assert cfg.transition_seed_offset == 777
    assert (cfg.full_latent_h, cfg.full_latent_w) == (45, 80)

    # The widget accepts 4-decimal A/beta (step/round 0.0001); the node must
    # forward them to the shared builder unrounded.
    four_digit = _capture_node_config(
        monkeypatch,
        noise_amplitude=12.1054,
        noise_decay_exponent=0.7732,
    )
    assert four_digit.noise_amplitude == 12.1054
    assert four_digit.noise_decay_exponent == 0.7732


def test_legacy_aliases_still_resolve(monkeypatch):
    assert PRESET_TO_STAGES["quarter_half_3q_full"] == 4
    assert _capture_node_config(monkeypatch, preset="quarter_half_3q_full").scales == STAGES_TO_SCALES[4]
    assert _capture_node_config(monkeypatch, delta=0.03).delta == pytest.approx(0.03)
    assert _capture_node_config(monkeypatch, **{"Tolerance (Delta)": 0.02}).delta == pytest.approx(0.02)
