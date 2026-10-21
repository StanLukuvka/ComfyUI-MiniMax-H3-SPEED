"""Public SAMPLER-input contracts for the three nodes."""

import importlib
import inspect

import pytest
import torch

from conftest import RecordingEchoGuider, StubSampler, make_fake_noise, make_latent


NODES = (
    ("sampler_node", "MiniMaxH3SPEEDSampler", "sample"),
    ("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual", "sample"),
    ("sampler_sigma_harvest_node", "MiniMaxH3HarvestToConfig", "harvest"),
)

#: SamplerCustomAdvanced input order, shared by all three nodes.
ADVANCED_INPUTS = ["noise", "guider", "sampler", "sigmas", "latent_image"]


def _node(module_name, class_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


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
    guider = RecordingEchoGuider()
    _node(module_name, class_name)().sample(
        noise=make_fake_noise(),
        guider=guider,
        sampler=sampler,
        sigmas=torch.linspace(1.0, 0.0, 11),
        latent_image=make_latent(h=8, w=8),
        **options,
    )
    assert len(guider.samplers) == stages
    assert all(seen is sampler for seen in guider.samplers)
