# FLOW-PRODUCED — Implementation Plan — Continuous SPEED Sigma Harvester.md §56 (commit 1) — flow-produced, do not hand-edit
"""Config-equivalence and schema regression tests for the shared automatic
SPEED config builder (speed_scripts/automatic_config.py).

Plan §56: the Automatic sampler path and build_automatic_speed_config must
produce equal SpeedConfig values from identical inputs. Plan §5/§7: the
Automatic node's INPUT_TYPES contract must not change.
"""
import importlib

import pytest
import torch

from conftest import install_comfy_stubs as _install_comfy_stubs
from conftest import make_nested

_install_comfy_stubs()


def _latent(full_h=45, full_w=80):
    """A ComfyUI LATENT dict shaped like the H3 packed AV latent."""
    video = torch.zeros(1, 1, 2, full_h, full_w)
    audio = torch.zeros(1, 1, 2, 44)
    return {"samples": make_nested(video, audio)}


def _node_config(latent, **overrides):
    """Build a config through the Automatic sampler node's sample() path.

    run_speed_pipeline is stubbed so no generation runs; the config the node
    built is captured from the call.
    """
    import sampler_node as node_mod

    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kw):
        captured["config"] = config
        return latent_image, latent_image

    node_mod = importlib.import_module("sampler_node")
    original = node_mod.run_speed_pipeline
    node_mod.run_speed_pipeline = fake_pipeline
    try:
        node = node_mod.MiniMaxH3SPEEDSampler()
        node.sample(
            noise=object(), guider=type("G", (), {})(), sigmas=None,
            latent_image=latent, **overrides,
        )
    finally:
        node_mod.run_speed_pipeline = original
    return captured["config"]


def test_input_schema_widgets_and_required_inputs():
    """Regression: the Automatic node's required inputs and widget defaults
    stay as shipped (calibration defaults may move; the schema shape may not)."""
    mod = importlib.import_module("sampler_node")
    required = mod.MiniMaxH3SPEEDSampler.INPUT_TYPES()["required"]
    for key in ("noise", "guider", "sigmas", "latent_image", "stages"):
        assert key in required, f"missing required input: {key}"
    assert "preset" not in required
    assert "transition_mode" not in required
    assert "Tolerance (Delta)" in required or "delta" in required
    assert "noise_amplitude" in required
    assert "noise_decay_exponent" in required
    assert "seed_offset" in required
    assert required["stages"] == ("INT", {"default": 3, "min": 2, "max": 4})
    assert required["noise_policy"] == (
        ["direct_coarse", "coupled_full_grid"], {"default": "direct_coarse"},
    )
    assert required["Tolerance (Delta)"] == (
        "FLOAT", {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001},
    )
    assert required["noise_amplitude"][1]["default"] == 12.105
    assert required["noise_decay_exponent"][1]["default"] == 0.773
    assert required["seed_offset"][1]["default"] == 10000


def test_node_builds_documented_config():
    """Plan §56: the node path must produce the documented SpeedConfig values.

    The expected values are written out as literals here — not read back from
    build_automatic_speed_config — so this test stays meaningful after the
    node delegates to the shared helper (both sides of a node-vs-helper
    comparison would otherwise be the same code).
    """
    node_cfg = _node_config(
        _latent(45, 80),
        stages=3,
        noise_policy="coupled_full_grid",
        delta=0.005,
        noise_amplitude=12.454,
        noise_decay_exponent=0.819,
        seed_offset=777,
    )
    assert node_cfg.scales == (0.3333333333, 0.6666666667, 1.0)
    assert node_cfg.transition_steps == (1, 2)
    assert node_cfg.transition_mode == "delta_custom"
    assert node_cfg.noise_policy == "coupled_full_grid"
    assert node_cfg.delta == 0.005
    assert node_cfg.noise_amplitude == 12.454
    assert node_cfg.noise_decay_exponent == 0.819
    assert node_cfg.transition_seed_offset == 777
    assert (node_cfg.full_latent_h, node_cfg.full_latent_w) == (45, 80)


def test_node_and_helper_build_identical_configs():
    """Plan §56: node path vs helper path, identical inputs -> identical config."""
    from speed_scripts.automatic_config import build_automatic_speed_config

    latent = _latent(45, 80)
    kwargs = dict(
        stages=3,
        noise_policy="coupled_full_grid",
        delta=0.005,
        noise_amplitude=12.454,
        noise_decay_exponent=0.819,
        seed_offset=777,
    )
    node_cfg = _node_config(latent, **kwargs)
    helper_cfg = build_automatic_speed_config(latent, **kwargs)
    for field in ("scales", "transition_steps", "transition_mode",
                  "noise_policy", "delta", "noise_amplitude",
                  "noise_decay_exponent", "transition_seed_offset",
                  "full_latent_h", "full_latent_w"):
        assert getattr(node_cfg, field) == getattr(helper_cfg, field), field


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_scales_ladder_matches_stages(stages):
    from speed_scripts.automatic_config import STAGES_TO_SCALES

    cfg = _node_config(_latent(), stages=stages)
    assert cfg.scales == STAGES_TO_SCALES[stages]
    assert cfg.transition_mode == "delta_custom"
    assert cfg.transition_steps == tuple(range(1, len(cfg.scales)))


def test_delta_alias_order_unchanged():
    """The Tolerance (Delta) alias chain must keep its priority order."""
    from speed_scripts.automatic_config import build_automatic_speed_config

    latent = _latent()
    via_widget = _node_config(latent, **{"Tolerance (Delta)": 0.02})
    assert via_widget.delta == 0.02
    via_old = _node_config(latent, delta=0.03)
    assert via_old.delta == 0.03
    assert via_old.delta == build_automatic_speed_config(
        latent, stages=3, noise_policy="direct_coarse", delta=0.03,
        noise_amplitude=7.394, noise_decay_exponent=0.62, seed_offset=10000,
    ).delta


def test_full_dims_from_latent():
    from speed_scripts.automatic_config import build_automatic_speed_config

    cfg = build_automatic_speed_config(
        _latent(45, 80), stages=2, noise_policy="direct_coarse", delta=0.01,
        noise_amplitude=7.394, noise_decay_exponent=0.62, seed_offset=10000,
    )
    assert (cfg.full_latent_h, cfg.full_latent_w) == (45, 80)
    cfg2 = build_automatic_speed_config(
        _latent(24, 40), stages=2, noise_policy="direct_coarse", delta=0.01,
        noise_amplitude=7.394, noise_decay_exponent=0.62, seed_offset=10000,
    )
    assert (cfg2.full_latent_h, cfg2.full_latent_w) == (24, 40)


def test_preset_alias_maps_to_stages():
    mod = importlib.import_module("sampler_node")
    from speed_scripts.automatic_config import PRESET_TO_STAGES

    assert PRESET_TO_STAGES["quarter_half_3q_full"] == 4
    # preset kwarg still routes through the alias on the node path
    cfg = _node_config(_latent(), preset="quarter_half_3q_full")
    assert cfg.scales == (0.25, 0.5, 0.75, 1.0)
