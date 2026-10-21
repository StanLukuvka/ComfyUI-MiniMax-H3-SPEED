"""Node registration and the Automatic and Manual SPEED node contracts."""

import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest
import torch

from conftest import (
    EchoGuider,
    EchoNoise,
    StubSampler,
    capture_node_config,
    make_latent,
)
from speed_scripts.flow import aligned_sigma


def _node(module_name, class_name):
    return getattr(importlib.import_module(module_name), class_name)


def _automatic():
    return _node("sampler_node", "MiniMaxH3SPEEDSampler")


def _manual():
    return _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")


NODES = (
    ("sampler_node", "MiniMaxH3SPEEDSampler", "sample"),
    ("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual", "sample"),
    ("sampler_sigma_harvest_node", "MiniMaxH3HarvestToConfig", "harvest"),
)

#: SamplerCustomAdvanced input order, shared by all three nodes.
ADVANCED_INPUTS = ["noise", "guider", "sampler", "sigmas", "latent_image"]


# ---------------------------------------------------------------------------
# Registration and inputs shared by all three nodes
# ---------------------------------------------------------------------------

def test_package_registers_all_shipped_nodes():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("speed_pack_test", root / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.NODE_CLASS_MAPPINGS) == {
        "MiniMaxH3SPEEDSampler",
        "MiniMaxH3SPEEDSamplerManual",
        "MiniMaxH3HarvestToConfig",
    }
    assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(module.NODE_CLASS_MAPPINGS)


def test_nodes_take_the_sampler_custom_advanced_inputs_in_order():
    for module_name, class_name, _method in NODES:
        required = _node(module_name, class_name).INPUT_TYPES()["required"]
        assert list(required)[:5] == ADVANCED_INPUTS, class_name
        assert required["sampler"] == ("SAMPLER",)


def test_programmatic_node_calls_take_the_sampler_custom_advanced_parameters_in_order():
    for module_name, class_name, method in NODES:
        function = getattr(_node(module_name, class_name), method)
        params = inspect.signature(function).parameters
        assert list(params)[1:6] == ADVANCED_INPUTS, class_name  # after self
        assert params["sampler"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("module_name", "class_name", "options", "stages"),
    (
        (
            "sampler_node", "MiniMaxH3SPEEDSampler",
            {"stages": 3, "noise_amplitude": 12.105, "noise_decay_exponent": 0.773},
            3,
        ),
        (
            "sampler_node_manual", "MiniMaxH3SPEEDSamplerManual",
            {
                "transition_goal_1": 3, "transition_resolution_1": .25,
                "transition_goal_2": 5, "transition_resolution_2": .5,
                "transition_goal_3": 8, "transition_resolution_3": .75,
                "transition_goal_4": 15, "transition_resolution_4": 1.0,
            },
            4,
        ),
    ),
    ids=("automatic", "manual"),
)
def test_speed_nodes_pass_the_given_sampler_to_every_stage(module_name, class_name, options, stages):
    sampler = StubSampler("heun")
    guider = EchoGuider()
    _node(module_name, class_name)().sample(
        noise=EchoNoise(),
        guider=guider,
        sampler=sampler,
        sigmas=torch.linspace(1.0, 0.0, 11),
        latent_image=make_latent(h=8, w=8),
        **options,
    )
    assert len(guider.samplers) == stages
    assert all(seen is sampler for seen in guider.samplers)


# ---------------------------------------------------------------------------
# Automatic
# ---------------------------------------------------------------------------

def test_automatic_node_public_surface_and_defaults():
    cls = _automatic()
    required = cls.INPUT_TYPES()["required"]
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert {"noise", "guider", "sigmas", "latent_image", "stages",
            "noise_policy", "Tolerance (Delta)", "noise_amplitude",
            "noise_decay_exponent", "seed_offset"} <= set(required)
    assert "transition_mode" not in required
    assert required["stages"] == ("INT", {"default": 3, "min": 2, "max": 4})
    assert required["Tolerance (Delta)"][1]["default"] == 0.005
    assert required["noise_amplitude"][1]["default"] == 12.105
    assert required["noise_decay_exponent"][1]["default"] == 0.773
    assert required["noise_amplitude"][1]["step"] == 0.0001
    assert required["noise_amplitude"][1]["round"] == 0.0001
    assert required["noise_decay_exponent"][1]["step"] == 0.0001
    assert required["noise_decay_exponent"][1]["round"] == 0.0001
    assert required["noise_amplitude"][1]["min"] > 0
    assert required["noise_decay_exponent"][1]["min"] > 0
    assert required["Tolerance (Delta)"][1]["step"] == 0.001


def test_node_forwards_generation_configuration_to_shared_builder(monkeypatch):
    from speed_scripts.planning import STAGES_TO_SCALES

    cfg = capture_node_config(
        monkeypatch,
        _automatic(),
        stages=3,
        noise_policy="coupled_full_grid",
        **{"Tolerance (Delta)": 0.007},
        noise_amplitude=13.5,
        noise_decay_exponent=0.9,
        seed_offset=777,
    )
    assert cfg.scales == STAGES_TO_SCALES[3]
    assert cfg.transition_mode == "delta_custom"
    assert cfg.noise_policy == "coupled_full_grid"
    assert (cfg.delta, cfg.noise_amplitude, cfg.noise_decay_exponent) == (0.007, 13.5, 0.9)
    assert cfg.transition_seed_offset == 777

    four_digit = capture_node_config(
        monkeypatch,
        _automatic(),
        noise_amplitude=12.1054,
        noise_decay_exponent=0.7732,
    )
    assert four_digit.noise_amplitude == 12.1054
    assert four_digit.noise_decay_exponent == 0.7732


def test_unknown_automatic_options_fail_closed(monkeypatch):
    with pytest.raises(TypeError, match="Unexpected Automatic option"):
        capture_node_config(monkeypatch, _automatic(), removed_option="old")


def test_tolerance_widget_value_is_forwarded(monkeypatch):
    assert capture_node_config(
        monkeypatch,
        _automatic(),
        **{"Tolerance (Delta)": 0.02},
    ).delta == pytest.approx(0.02)


def test_removed_tolerance_alias_fails_closed(monkeypatch):
    with pytest.raises(TypeError, match="Unexpected Automatic option"):
        capture_node_config(monkeypatch, _automatic(), delta=0.03)


@pytest.mark.parametrize("stages", [1, 5, 0, True, 3.0, "3"])
def test_automatic_rejects_invalid_stage_counts(monkeypatch, stages):
    with pytest.raises(ValueError, match="stages must be exactly 2, 3, or 4"):
        capture_node_config(monkeypatch, _automatic(), stages=stages)


# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------

def _run_manual(sigmas, **kwargs):
    kwargs.setdefault("sampler", StubSampler())
    guider = EchoGuider()
    output = _manual()().sample(
        noise=EchoNoise(),
        guider=guider,
        sigmas=sigmas,
        latent_image=make_latent(h=8, w=8),
        **kwargs,
    )
    return output, guider.sigma_calls, guider.latent_sizes


def test_manual_node_public_surface():
    cls = _manual()
    required = cls.INPUT_TYPES()["required"]
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert {"noise", "guider", "sigmas", "latent_image", "ratio_mode",
            "transition_goal_1", "transition_resolution_1",
            "transition_goal_4", "transition_resolution_4"} <= set(required)
    assert required["transition_resolution_1"][1]["default"] == 0.25
    assert required["transition_resolution_4"][1]["default"] == 1.0


def test_manual_config_stays_explicit_with_inherited_noise_fit(monkeypatch):
    cfg = capture_node_config(monkeypatch, _manual())
    # Manual uses explicit boundaries, so A and beta do not affect planning.
    assert cfg.transition_mode == "explicit"
    assert cfg.delta == 0.01
    assert (cfg.noise_amplitude, cfg.noise_decay_exponent) == (12.105, 0.773)


def test_steps_mode_executes_global_boundaries_and_alignment():
    sigmas = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])
    (_out, _denoised), calls, shapes = _run_manual(
        sigmas,
        ratio_mode="steps",
        transition_goal_1=3, transition_resolution_1=.25,
        transition_goal_2=5, transition_resolution_2=.5,
        transition_goal_3=8, transition_resolution_3=.75,
        transition_goal_4=15, transition_resolution_4=1.0,
    )

    original = [float(s) for s in sigmas]
    expected = [original[:4]]
    _, q1 = aligned_sigma(original[3], 2.0)
    expected.append([q1, *original[4:6]])
    _, q2 = aligned_sigma(original[5], 1.5)
    expected.append([q2, *original[6:9]])
    _, q3 = aligned_sigma(original[8], 4.0 / 3.0)
    expected.append([q3, *original[9:]])

    assert len(calls) == 4
    for actual, wanted in zip(calls, expected):
        assert actual == pytest.approx(wanted)
    assert sum(len(call) - 1 for call in calls) == len(sigmas) - 1
    assert shapes == [(2, 2), (4, 4), (6, 6), (8, 8)]


def test_shortest_four_stage_schedule_is_valid():
    sigmas = torch.linspace(1.0, 0.0, 5)
    (_out, _denoised), calls, _shapes = _run_manual(
        sigmas,
        transition_goal_1=1, transition_resolution_1=.25,
        transition_goal_2=2, transition_resolution_2=.5,
        transition_goal_3=3, transition_resolution_3=.75,
        transition_goal_4=15, transition_resolution_4=1.0,
    )
    assert [len(call) for call in calls] == [2, 2, 2, 2]
    assert sum(len(call) - 1 for call in calls) == 4


def test_ratio_mode_uses_goal_only_for_boundary_position():
    sigmas = torch.linspace(1.0, 0.0, 11)
    (_out, _denoised), calls, shapes = _run_manual(
        sigmas,
        ratio_mode="ratio",
        transition_goal_1=.3, transition_resolution_1=.25,
        transition_goal_2=.6, transition_resolution_2=.5,
        transition_goal_3=0, transition_resolution_3=.75,
        # The final goal is ignored.
        transition_goal_4=15, transition_resolution_4=1.0,
    )
    assert [len(call) for call in calls] == [4, 4, 5]  # boundaries 3, 6
    assert shapes == [(2, 2), (4, 4), (8, 8)]


def test_disabled_intermediate_stage_shifts_ladder_down():
    sigmas = torch.linspace(1.0, 0.0, 11)
    (_out, _denoised), calls, shapes = _run_manual(
        sigmas,
        transition_goal_1=3, transition_resolution_1=.25,
        transition_goal_2=5, transition_resolution_2=.5,
        transition_goal_3=8, transition_resolution_3=0,
        transition_goal_4=15, transition_resolution_4=1.0,
    )
    assert len(calls) == 3
    assert shapes == [(2, 2), (4, 4), (8, 8)]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (
            dict(ratio_mode="steps", transition_goal_2=5.5),
            "whole step index",
        ),
        (
            dict(
                ratio_mode="ratio",
                transition_goal_1=1.1,
                transition_resolution_1=.25,
                transition_goal_2=.6,
                transition_resolution_2=.5,
                transition_goal_3=0,
                transition_resolution_3=0,
                transition_goal_4=15,
                transition_resolution_4=1.0,
            ),
            "fraction of the schedule",
        ),
        (
            dict(
                transition_goal_2=0,
                transition_goal_3=0,
                transition_goal_4=0,
            ),
            "at least two active stages",
        ),
    ],
)
def test_manual_rejects_ambiguous_or_incomplete_schedules(kwargs, error):
    with pytest.raises(ValueError, match=error):
        _run_manual(torch.linspace(1.0, 0.0, 20), **kwargs)
