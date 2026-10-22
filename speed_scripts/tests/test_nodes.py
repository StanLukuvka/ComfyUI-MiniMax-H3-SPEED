"""The three nodes: registration, inputs and defaults, configs, schedules, and rejected inputs."""

import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from conftest import EchoGuider, EchoNoise, StubSampler, expected_stage_sigmas, make_latent

AUTOMATIC = ("sampler_node", "MiniMaxH3SPEEDSampler")
MANUAL = ("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
HARVEST = ("sampler_sigma_harvest_node", "MiniMaxH3HarvestToConfig")
POLICIES = ["direct_coarse", "coupled_full_grid"]


def capture_node_config(monkeypatch, node_cls, **inputs):
    """Call a SPEED node with the pipeline stubbed out; return the SpeedConfig it built."""
    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kwargs):
        captured["config"] = config
        return latent_image, latent_image

    monkeypatch.setattr(importlib.import_module(node_cls.__module__), "run_speed_pipeline", fake_pipeline)
    inputs.setdefault("sampler", StubSampler())
    inputs.setdefault("sigmas", torch.linspace(1.0, 0.0, 11))
    node_cls().sample(noise=EchoNoise(), guider=EchoGuider(), latent_image=make_latent(), **inputs)
    return captured["config"]


def _node(module_and_class):
    module_name, class_name = module_and_class
    return getattr(importlib.import_module(module_name), class_name)


def test_package_registers_exactly_the_three_nodes():
    spec = importlib.util.spec_from_file_location(
        "speed_pack_test", Path(__file__).resolve().parents[2] / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3SPEEDSampler", "MiniMaxH3SPEEDSamplerManual", "MiniMaxH3HarvestToConfig",
    }
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(module.NODE_CLASS_MAPPINGS)


@pytest.mark.parametrize("node", [AUTOMATIC, MANUAL, HARVEST], ids=lambda n: n[1])
def test_nodes_take_the_sampler_custom_advanced_inputs(node):
    cls = _node(node)
    advanced = ["noise", "guider", "sampler", "sigmas", "latent_image"]
    assert list(cls.INPUT_TYPES()["required"])[:5] == advanced
    assert cls.INPUT_TYPES()["required"]["sampler"][0] == "SAMPLER"
    params = inspect.signature(getattr(cls, cls.FUNCTION)).parameters
    assert list(params)[1:6] == advanced  # after self
    assert params["sampler"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    "node,group,name,expected",
    [
        # Optional on every node, so old workflows and API prompts without it keep running H3.
        (AUTOMATIC, "optional", "model_type", (["h3_video", "image"], {"default": "h3_video"})),
        (MANUAL, "optional", "model_type", (["h3_video", "image"], {"default": "h3_video"})),
        (HARVEST, "optional", "model_type", (["h3_video", "image"], {"default": "h3_video"})),
        (AUTOMATIC, "required", "noise_policy", (POLICIES, {"default": "direct_coarse"})),
        (MANUAL, "required", "noise_policy", (POLICIES, {"default": "direct_coarse"})),
        (AUTOMATIC, "required", "stages", ("INT", {"default": 3, "min": 2, "max": 4})),
        # Baked Euler calibration: Δ0.005 A12.105 β0.773, editable to four decimals.
        (AUTOMATIC, "required", "Tolerance (Delta)",
         ("FLOAT", {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001})),
        (AUTOMATIC, "required", "noise_amplitude",
         ("FLOAT", {"default": 12.105, "min": 0.0001, "max": 1e6, "step": 0.0001, "round": 0.0001})),
        (AUTOMATIC, "required", "noise_decay_exponent",
         ("FLOAT", {"default": 0.773, "min": 0.0001, "max": 10.0, "step": 0.0001, "round": 0.0001})),
        (HARVEST, "optional", "Tolerance (Delta)",
         ("FLOAT", {"default": 0.005, "min": 1e-4, "max": 0.5, "step": 0.001})),
        (MANUAL, "required", "ratio_mode", (["steps", "ratio"], {"default": "steps"})),
        (MANUAL, "required", "transition_resolution_1", ("FLOAT", {"default": 0.25, "min": 0, "max": 1})),
        (MANUAL, "required", "transition_resolution_4", ("FLOAT", {"default": 1.0, "min": 0, "max": 1})),
    ],
)
def test_node_input_defaults(node, group, name, expected):
    kind, options = _node(node).INPUT_TYPES()[group][name]
    assert (kind, {k: v for k, v in options.items() if k != "tooltip"}) == expected


@pytest.mark.parametrize("node", [AUTOMATIC, MANUAL, HARVEST], ids=lambda n: n[1])
def test_every_input_and_output_has_a_tooltip(node):
    cls = _node(node)
    for section in cls.INPUT_TYPES().values():
        for name, spec in section.items():
            assert spec[1].get("tooltip", "").strip(), name
    assert len(cls.OUTPUT_TOOLTIPS) == len(cls.RETURN_TYPES)
    assert all(tip.strip() for tip in cls.OUTPUT_TOOLTIPS)
    assert cls.DESCRIPTION.strip()


@pytest.mark.parametrize("stages,scales", [(2, (.5, 1.0)), (3, (1 / 3, 2 / 3, 1.0)), (4, (.25, .5, .75, 1.0))])
def test_automatic_builds_a_delta_custom_config_from_its_inputs(monkeypatch, stages, scales):
    cfg = capture_node_config(
        monkeypatch, _node(AUTOMATIC), stages=stages, noise_policy="coupled_full_grid",
        **{"Tolerance (Delta)": 0.007}, noise_amplitude=12.1054, noise_decay_exponent=0.7732, seed_offset=777,
    )
    assert cfg.scales == pytest.approx(scales)
    # No placeholder boundaries: delta_custom places them from the live schedule and geometry.
    assert (cfg.transition_mode, cfg.transition_steps) == ("delta_custom", ())
    assert cfg.noise_policy == "coupled_full_grid"
    assert (cfg.delta, cfg.noise_amplitude, cfg.noise_decay_exponent) == (0.007, 12.1054, 0.7732)
    assert cfg.transition_seed_offset == 777


@pytest.mark.parametrize(
    "kwargs,error",
    [
        *[({"stages": bad}, "stages must be exactly 2, 3, or 4") for bad in (1, 5, 0, True, 3.0, "3")],
        # Unknown options fail closed, including the removed ``delta`` Tolerance alias.
        ({"removed_option": "old"}, "Unexpected Automatic option"),
        ({"delta": 0.03}, "Unexpected Automatic option"),
        ({"noise_policy": "bad"}, "unsupported noise_policy"),
        ({"Tolerance (Delta)": 1.0}, r"delta must be in \(0, 1\)"),
        *[({field: bad}, "positive finite")
          for field in ("noise_amplitude", "noise_decay_exponent")
          for bad in (float("nan"), float("inf"), 0.0, -1.0)],
    ],
)
def test_automatic_rejects_bad_inputs(monkeypatch, kwargs, error):
    with pytest.raises((TypeError, ValueError), match=error):
        capture_node_config(monkeypatch, _node(AUTOMATIC), **kwargs)


def _stages(*pairs):
    """Manual inputs from up to four ``(goal, resolution)`` pairs."""
    out = {}
    for i, (goal, resolution) in enumerate(pairs, 1):
        out[f"transition_goal_{i}"], out[f"transition_resolution_{i}"] = goal, resolution
    return out


QUARTERS = (.25, .5, .75, 1.0)


@pytest.mark.parametrize(
    "steps,kwargs,boundaries,scales",
    [
        # The defaults: steps mode, goals 3/5/8 at quarter resolutions.
        (10, {}, (3, 5, 8), QUARTERS),
        (4, _stages((1, .25), (2, .5), (3, .75), (15, 1.0)), (1, 2, 3), QUARTERS),
        # Ratio goals are fractions of the schedule; the final goal is ignored.
        (10, {"ratio_mode": "ratio", **_stages((.3, .25), (.6, .5), (0, .75), (15, 1.0))},
         (3, 6), (.25, .5, 1.0)),
        # A zero resolution disables that stage and the ladder shifts down.
        (10, _stages((3, .25), (5, .5), (8, 0), (15, 1.0)), (3, 5), (.25, .5, 1.0)),
    ],
    ids=["defaults", "shortest", "ratio", "disabled_stage"],
)
def test_manual_runs_its_schedule_with_the_given_sampler(steps, kwargs, boundaries, scales):
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    sampler = StubSampler("heun")
    guider = EchoGuider()
    # No model_type: old workflows keep running H3.
    _node(MANUAL)().sample(
        noise=EchoNoise(), guider=guider, sampler=sampler, sigmas=sigmas,
        latent_image=make_latent(), **kwargs,
    )
    for actual, wanted in zip(guider.sigma_calls, expected_stage_sigmas(sigmas, boundaries, scales), strict=True):
        assert actual == pytest.approx(wanted)
    assert guider.latent_sizes == [(round(8 * s),) * 2 for s in scales]
    assert all(seen is sampler for seen in guider.samplers)


def test_automatic_runs_every_stage_with_the_given_sampler():
    sampler = StubSampler("heun")
    guider = EchoGuider()
    # No model_type: old workflows keep running H3.
    out, _ = _node(AUTOMATIC)().sample(
        noise=EchoNoise(), guider=guider, sampler=sampler, sigmas=torch.linspace(1.0, 0.0, 11),
        latent_image=make_latent(), stages=3,
    )
    assert guider.latent_sizes == [(3, 3), (5, 5), (8, 8)]
    assert all(seen is sampler for seen in guider.samplers) and len(guider.samplers) == 3
    assert tuple(out["samples"].unbind()[0].shape[-2:]) == (8, 8)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"ratio_mode": "steps", "transition_goal_2": 5.5}, "whole step index"),
        ({"ratio_mode": "ratio", **_stages((1.1, .25), (.6, .5), (0, 0), (15, 1.0))}, "fraction of the schedule"),
        (_stages((3, .25), (0, .5), (0, .75), (0, 1.0)), "at least two active stages"),
        (_stages((3, .25), (3, .5), (8, .75), (15, 1.0)), "strictly increasing"),
        (_stages((3, .25), (25, .5), (8, .75), (15, 1.0)), "interior step indices"),
        (_stages((3, .5), (5, .5), (8, .75), (15, 1.0)), "scales must be strictly increasing"),
        (_stages((3, .25), (5, .5), (8, .75), (15, .9)), "final scale must be 1.0"),
        ({"ratio_mode": "fraction"}, "unsupported ratio_mode"),
    ],
)
def test_manual_rejects_bad_schedules(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _node(MANUAL)().sample(
            noise=EchoNoise(), guider=EchoGuider(), sampler=StubSampler(),
            sigmas=torch.linspace(1.0, 0.0, 20), latent_image=make_latent(), **kwargs,
        )
