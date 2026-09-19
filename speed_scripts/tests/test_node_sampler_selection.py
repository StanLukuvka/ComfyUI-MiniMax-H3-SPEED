"""Public sampler-selection contracts for the V2 nodes."""

import importlib
import inspect

import torch

from conftest import make_fake_noise, make_latent
import speed_scripts.h3_runtime as h3_runtime
from speed_scripts.res_multistep_adapter import ResMultistepSampler
from speed_scripts.sampler_support import (
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
    create_speed_sampler_handle,
)


def _node(module_name, class_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


class SamplerRecordingGuider:
    """Guider that records the sampler object used by every stage call."""

    class Model:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

        def process_latent_out(self, x):
            return x

    def __init__(self):
        self.model_patcher = type("P", (), {"model": self.Model()})()
        self.samplers = []

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        count = len(sigmas) - 1
        if callback is not None:
            for i in range(count):
                callback(i, latent_image, latent_image, count)
        return latent_image


def _sigmas():
    return torch.linspace(1.0, 0.0, 11)


def _run_automatic(sigmas, sampler_name, **kwargs):
    cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    guider = SamplerRecordingGuider()
    output = cls().sample(
        make_fake_noise(),
        guider,
        sigmas,
        make_latent(h=8, w=8),
        sampler_name=sampler_name,
        stages=3,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        **kwargs,
    )
    return output, guider.samplers


def _run_manual(sigmas, sampler_name, **kwargs):
    cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    guider = SamplerRecordingGuider()
    output = cls().sample(
        make_fake_noise(),
        guider,
        sigmas,
        make_latent(h=8, w=8),
        sampler_name=sampler_name,
        transition_goal_1=3,
        transition_resolution_1=.25,
        transition_goal_2=5,
        transition_resolution_2=.5,
        transition_goal_3=8,
        transition_resolution_3=.75,
        transition_goal_4=15,
        transition_resolution_4=1.0,
        **kwargs,
    )
    return output, guider.samplers


def test_sampler_dropdowns_use_exact_supported_names_and_default_to_euler():
    automatic_cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    manual_cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    harvest_cls = _node("sampler_sigma_harvest_node", "MiniMaxH3HarvestToConfig")

    for cls in (automatic_cls, manual_cls, harvest_cls):
        values, options = cls.INPUT_TYPES()["required"]["sampler_name"]
        assert tuple(values) == tuple(SUPPORTED_SPEED_SAMPLERS)
        assert options["default"] == "euler"


def test_programmatic_node_calls_require_sampler_name():
    automatic = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    manual = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    harvest = _node("sampler_sigma_harvest_node", "MiniMaxH3HarvestToConfig")

    for function in (automatic.sample, manual.sample, harvest.harvest):
        assert inspect.signature(function).parameters["sampler_name"].default is inspect.Parameter.empty


def test_automatic_passes_non_default_name_to_runtime():
    _out, samplers = _run_automatic(_sigmas(), "heun")
    assert samplers
    assert all(sampler == ("sampler", "heun") for sampler in samplers)


def test_manual_passes_non_default_name_to_runtime():
    _out, samplers = _run_manual(_sigmas(), "heun")
    assert samplers
    assert all(sampler == ("sampler", "heun") for sampler in samplers)


def _run_res_node(node_runner, monkeypatch):
    captured = []

    def factory(name, **factory_kwargs):
        handle = create_speed_sampler_handle(name, **factory_kwargs)
        captured.append((name, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    _out, samplers = node_runner(_sigmas(), "res_multistep")
    return captured, samplers


def test_automatic_routes_res_to_stateful_public_factory(monkeypatch):
    captured, samplers = _run_res_node(_run_automatic, monkeypatch)
    assert len(captured) == 1
    name, handle = captured[0]
    assert name == "res_multistep"
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    assert samplers == [samplers[0]] * 3
    assert isinstance(samplers[0], ResMultistepSampler)


def test_manual_routes_res_to_stateful_public_factory(monkeypatch):
    captured, samplers = _run_res_node(_run_manual, monkeypatch)
    assert len(captured) == 1
    name, handle = captured[0]
    assert name == "res_multistep"
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    assert samplers == [samplers[0]] * 4
    assert isinstance(samplers[0], ResMultistepSampler)
