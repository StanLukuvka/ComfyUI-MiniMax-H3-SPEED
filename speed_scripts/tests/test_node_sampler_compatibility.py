"""Public-input compatibility for the sampler selectors (plan §3, §9).

Old workflows have no sampler field: both nodes must execute as Euler, keep
every pre-existing input in its original position, and append the new
selectors last with signature defaults of ``"euler"`` and ``"reset"``.
"""

# FLOW-PRODUCED: V2 sampler widget compatibility coverage.

import inspect
import importlib

import torch

from conftest import make_fake_noise, make_latent
import speed_scripts.h3_runtime as h3_runtime
from speed_scripts.res_multistep_adapter import ResMultistepSampler
from speed_scripts.sampler_support import (
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
    create_speed_sampler_handle,
)


def _same_latents(a, b):
    """Compare two latent dicts by tensor value, not object identity."""
    for part_a, part_b in zip(a["samples"].unbind(), b["samples"].unbind()):
        if not torch.equal(part_a, part_b):
            return False
    return True


# The widget order before the dropdowns existed. The selectors must be appended
# after it, never inserted between existing widgets.
AUTOMATIC_INPUT_ORDER_BEFORE = (
    "noise",
    "guider",
    "sigmas",
    "latent_image",
    "stages",
    "noise_policy",
    "Tolerance (Delta)",
    "noise_amplitude",
    "noise_decay_exponent",
    "seed_offset",
)

MANUAL_INPUT_ORDER_BEFORE = (
    "noise",
    "guider",
    "sigmas",
    "latent_image",
    "noise_policy",
    "seed_offset",
    "ratio_mode",
    "transition_goal_1",
    "transition_resolution_1",
    "transition_goal_2",
    "transition_resolution_2",
    "transition_goal_3",
    "transition_resolution_3",
    "transition_goal_4",
    "transition_resolution_4",
)


def _node(module_name, class_name):
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


class SamplerRecordingGuider:
    """Guider that records the native sampler object every stage call uses."""

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


def _automatic_kwargs():
    return dict(
        stages=3,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
    )


def _run_automatic(sigmas, **kwargs):
    cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    guider = SamplerRecordingGuider()
    output = cls().sample(
        make_fake_noise(),
        guider,
        sigmas,
        make_latent(h=8, w=8),
        **kwargs,
    )
    return output, guider.samplers


def _run_manual(sigmas, **kwargs):
    cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    guider = SamplerRecordingGuider()
    output = cls().sample(
        make_fake_noise(),
        guider,
        sigmas,
        make_latent(h=8, w=8),
        transition_goal_1=3, transition_resolution_1=.25,
        transition_goal_2=5, transition_resolution_2=.5,
        transition_goal_3=8, transition_resolution_3=.75,
        transition_goal_4=15, transition_resolution_4=1.0,
        **kwargs,
    )
    return output, guider.samplers


def _sigmas():
    return torch.linspace(1.0, 0.0, 11)


# ---------------------------------------------------------------------------
# Old-workflow payloads: no sampler_name anywhere (plan §3 "Compatibility
# test", §9 "Old no-name/default path selects Euler")
# ---------------------------------------------------------------------------

def test_automatic_without_sampler_name_executes_euler():
    sigmas = _sigmas()
    no_field, samplers = _run_automatic(sigmas, **_automatic_kwargs())
    explicit, _ = _run_automatic(
        sigmas, sampler_name="euler", **_automatic_kwargs()
    )
    assert samplers, "run never reached the guider"
    assert all(sampler == ("sampler", "euler") for sampler in samplers)
    assert _same_latents(no_field[0], explicit[0])


def test_manual_without_sampler_name_executes_euler():
    sigmas = _sigmas()
    no_field, samplers = _run_manual(sigmas)
    explicit, _ = _run_manual(sigmas, sampler_name="euler")
    assert samplers, "run never reached the guider"
    assert all(sampler == ("sampler", "euler") for sampler in samplers)
    assert _same_latents(no_field[0], explicit[0])


def test_sample_signatures_default_to_euler():
    automatic_cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    manual_cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    for cls in (automatic_cls, manual_cls):
        param = inspect.signature(cls.sample).parameters["sampler_name"]
        assert param.default == "euler"


# ---------------------------------------------------------------------------
# Widget/input ordering (plan §3: "old widget/input ordering is not changed
# before the new field")
# ---------------------------------------------------------------------------

def test_automatic_appends_selector_after_existing_inputs():
    cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    required = cls.INPUT_TYPES()["required"]
    keys = tuple(required)
    assert keys == AUTOMATIC_INPUT_ORDER_BEFORE + ("sampler_name", "res_history_mode")


def test_manual_appends_selector_after_existing_inputs():
    cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    required = cls.INPUT_TYPES()["required"]
    keys = tuple(required)
    assert keys == MANUAL_INPUT_ORDER_BEFORE + ("sampler_name", "res_history_mode")


def test_sampler_dropdown_is_exactly_the_supported_names_defaulting_to_euler():
    automatic_cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    manual_cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    for cls in (automatic_cls, manual_cls):
        values, options = cls.INPUT_TYPES()["required"]["sampler_name"]
        assert tuple(values) == tuple(SUPPORTED_SPEED_SAMPLERS)
        assert options["default"] == "euler"


def test_history_dropdown_is_exactly_reset_and_projected_defaulting_to_reset():
    automatic_cls = _node("sampler_node", "MiniMaxH3SPEEDSampler")
    manual_cls = _node("sampler_node_manual", "MiniMaxH3SPEEDSamplerManual")
    for cls in (automatic_cls, manual_cls):
        values, options = cls.INPUT_TYPES()["required"]["res_history_mode"]
        assert tuple(values) == ("reset", "projected")
        assert options["default"] == "reset"


# ---------------------------------------------------------------------------
# Selector pass-through (plan §3: the dropdown feeds run_speed_pipeline, so a
# non-default selection must reach the runtime handle unchanged)
# ---------------------------------------------------------------------------

def test_automatic_passes_non_default_name_to_the_runtime():
    _out, samplers = _run_automatic(
        _sigmas(), sampler_name="heun", **_automatic_kwargs()
    )
    assert samplers, "run never reached the guider"
    assert all(sampler == ("sampler", "heun") for sampler in samplers)


def test_manual_passes_non_default_name_to_the_runtime():
    _out, samplers = _run_manual(_sigmas(), sampler_name="heun")
    assert samplers, "run never reached the guider"
    assert all(sampler == ("sampler", "heun") for sampler in samplers)


def _run_res_node(node_runner, monkeypatch, **kwargs):
    captured = []

    def factory(name, **kwargs):
        handle = create_speed_sampler_handle(name, **kwargs)
        captured.append((name, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    _out, samplers = node_runner(
        _sigmas(), sampler_name="res_multistep", **kwargs
    )
    return captured, samplers


def test_automatic_routes_res_to_the_stateful_public_factory(monkeypatch):
    captured, samplers = _run_res_node(_run_automatic, monkeypatch)
    assert len(captured) == 1
    name, handle = captured[0]
    assert name == "res_multistep"
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    assert samplers == [samplers[0]] * 3
    assert isinstance(samplers[0], ResMultistepSampler)


def test_manual_routes_res_to_the_stateful_public_factory(monkeypatch):
    captured, samplers = _run_res_node(_run_manual, monkeypatch)
    assert len(captured) == 1
    name, handle = captured[0]
    assert name == "res_multistep"
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    assert samplers == [samplers[0]] * 4
    assert isinstance(samplers[0], ResMultistepSampler)


def test_automatic_forwards_history_mode_to_the_runtime(monkeypatch):
    captured, _samplers = _run_res_node(
        _run_automatic, monkeypatch, res_history_mode="projected"
    )
    assert len(captured) == 1
    assert captured[0][1].history_mode == "projected"


def test_manual_forwards_history_mode_to_the_runtime(monkeypatch):
    captured, _samplers = _run_res_node(
        _run_manual, monkeypatch, res_history_mode="projected"
    )
    assert len(captured) == 1
    assert captured[0][1].history_mode == "projected"
