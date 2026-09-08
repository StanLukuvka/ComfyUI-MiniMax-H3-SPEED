"""Runtime-observer contracts for the multi-stage SPEED pipeline."""

import pytest
import torch

from conftest import make_fake_noise, make_latent, make_recording_guider
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import run_speed_pipeline


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])
FOUR_STAGE = SpeedConfig(
    scales=(.25, .5, .75, 1.0),
    transition_steps=(3, 5, 8),
    transition_mode="explicit",
)


class Observer:
    def __init__(self):
        self.start = []
        self.steps = []
        self.transitions = []
        self.end = []

    def on_run_start(self, event):
        self.start.append(event)

    def on_step(self, event, x0, x):
        self.steps.append((event, x0, x))

    def on_transition(self, event):
        self.transitions.append(event)

    def on_run_end(self, event):
        self.end.append(event)


def _run(config, observer=None, sigmas=SIGMAS, latent=None):
    calls = []
    guider = make_recording_guider(sigma_calls=calls)
    out, denoised = run_speed_pipeline(
        make_fake_noise(),
        guider,
        sigmas,
        latent or make_latent(),
        config,
        sampler=object(),
        disable_pbar=True,
        observer=observer,
    )
    return calls, out, denoised


def test_observer_reports_the_complete_global_event_stream():
    observer = Observer()
    _run(FOUR_STAGE, observer)

    assert len(observer.start) == len(observer.end) == 1
    assert observer.start[0].transition_steps == (3, 5, 8)
    assert observer.start[0].global_steps == 10
    assert observer.end[0].transition_count == 3

    events = [event for event, _x0, _x in observer.steps]
    assert [event.callback_index for event in events] == list(range(10))
    assert [event.global_schedule_index for event in events] == list(range(10))
    assert [event.stage_index for event in events] == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3]

    original = [float(s) for s in SIGMAS]
    for transition_index, boundary in enumerate((3, 5, 8)):
        transition = observer.transitions[transition_index]
        ratio = FOUR_STAGE.scales[transition_index + 1] / FOUR_STAGE.scales[transition_index]
        _, aligned = aligned_sigma(original[boundary], ratio)
        first_after = next(e for e in events if e.global_schedule_index == boundary)
        assert transition.global_schedule_index == boundary
        assert transition.sigma_before_alignment == pytest.approx(original[boundary])
        assert transition.sigma_after_alignment == pytest.approx(aligned)
        assert first_after.original_sigma == pytest.approx(original[boundary])
        assert first_after.actual_sigma == pytest.approx(aligned)


def test_observer_preserves_coincident_transition_semantics():
    sigmas = torch.linspace(1.0, 0.0, 11)
    cfg = SpeedConfig(
        scales=(.25, .5, 1.0),
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=8,
        full_latent_w=8,
    )
    observer = Observer()
    calls, _out, _denoised = _run(cfg, observer, sigmas, make_latent())

    assert [t.global_schedule_index for t in observer.transitions] == [1, 1]
    assert len(calls[1]) == 1
    assert not [e for e, _x0, _x in observer.steps if e.stage_index == 1]

    original = float(sigmas[1])
    _, once = aligned_sigma(original, 2.0)
    _, twice = aligned_sigma(once, 2.0)
    assert observer.transitions[0].sigma_after_alignment == pytest.approx(once)
    assert observer.transitions[1].sigma_before_alignment == pytest.approx(once)
    assert observer.transitions[1].sigma_after_alignment == pytest.approx(twice)


def test_attaching_observer_does_not_change_generation():
    plain_calls, plain_out, plain_denoised = _run(FOUR_STAGE)
    observer = Observer()
    observed_calls, observed_out, observed_denoised = _run(FOUR_STAGE, observer)

    assert plain_calls == observed_calls
    for plain, observed in zip(plain_out["samples"].unbind(), observed_out["samples"].unbind()):
        assert torch.equal(plain, observed)
    for plain, observed in zip(
        plain_denoised["samples"].unbind(), observed_denoised["samples"].unbind()
    ):
        assert torch.equal(plain, observed)
    assert len(observer.steps) == len(SIGMAS) - 1
