# FLOW-PRODUCED — Implementation Plan — Continuous SPEED Sigma Harvester.md §50-54 (commit 2) — flow-produced, do not hand-edit
"""Runtime observer hook tests.

Runs `run_speed_pipeline` with the same fake guider/noise infrastructure as
`test_global_transitions.py` and asserts what an observer sees: one step
event per actual denoising interval, one transition event per resolved
transition (coincident included), actual-vs-original sigma semantics, and
that attaching an observer leaves the generation bit-for-bit unchanged.
"""

import pytest
import torch

from conftest import install_comfy_stubs
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.observer import (
    SpeedRunEndEvent,
    SpeedRunStartEvent,
    SpeedStepEvent,
    SpeedTransitionEvent,
)

install_comfy_stubs()

SIGMAS_10 = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

FOUR_STAGE_CFG = SpeedConfig(
    scales=(0.25, 0.5, 0.75, 1.0),
    transition_steps=(3, 5, 8),
    transition_mode="explicit",
)


class RecordingGuider:
    """Fake guider recording every sigma schedule (same as test_global_transitions)."""

    def __init__(self):
        self.sigma_calls = []
        self.model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        self.conds = {}

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.sigma_calls.append([float(s) for s in sigmas])
        if callback is not None:
            for i in range(len(sigmas) - 1):
                callback(i, latent_image, latent_image, len(sigmas) - 1)
        return latent_image


class RecordingNoise:
    """Noise that regenerates the (video, audio) streams unchanged."""

    seed = 42

    def generate_noise(self, latent):
        samples = latent.get("samples")
        if getattr(samples, "is_nested", False):
            vids = [s for s in samples.unbind() if s.ndim == 5]
            auds = [s for s in samples.unbind() if s.ndim != 5]
            return type("NT", (), {"is_nested": True, "unbind": lambda self: vids + auds})()
        return samples


def make_latent(t=2, h=8, w=8):
    video = torch.zeros(1, 1, t, h, w)
    audio = torch.zeros(1, 1, 2, 44)
    nested = type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()
    return {"samples": nested}


class RecordingObserver:
    """Observer that records every event and callback payload it receives."""

    def __init__(self):
        self.run_start_events = []
        self.step_events = []
        self.transition_events = []
        self.run_end_events = []
        self.step_payloads = []

    def on_run_start(self, event):
        assert isinstance(event, SpeedRunStartEvent)
        self.run_start_events.append(event)

    def on_step(self, event, x0, x):
        assert isinstance(event, SpeedStepEvent)
        self.step_events.append(event)
        self.step_payloads.append((x0, x))

    def on_transition(self, event):
        assert isinstance(event, SpeedTransitionEvent)
        self.transition_events.append(event)

    def on_run_end(self, event):
        assert isinstance(event, SpeedRunEndEvent)
        self.run_end_events.append(event)

    @property
    def global_indices(self):
        return [e.global_schedule_index for e in self.step_events]


def run_pipeline(cfg, observer, sigmas=SIGMAS_10, latent=None):
    guider = RecordingGuider()
    out, denoised = run_speed_pipeline(
        RecordingNoise(), guider, sigmas,
        latent if latent is not None else make_latent(), cfg,
        sampler=type("S", (), {"name": "euler"})(),
        disable_pbar=True,
        observer=observer,
    )
    return guider, out, denoised


def test_step_and_transition_event_counts():
    """§50: 4-stage (3,5,8) on 11 sigmas -> 10 step events, 3 transitions.

    Global denoising indices 0..2, 3..4, 5..7, 8..9; each exactly once.
    """
    observer = RecordingObserver()
    run_pipeline(FOUR_STAGE_CFG, observer)

    assert observer.global_indices == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert len(observer.transition_events) == 3
    assert len(observer.run_start_events) == 1
    assert len(observer.run_end_events) == 1
    assert len(observer.step_payloads) == 10

    start = observer.run_start_events[0]
    assert start.n_stages == 4
    assert start.scales == (0.25, 0.5, 0.75, 1.0)
    assert start.transition_steps == (3, 5, 8)
    assert start.global_steps == 10

    end = observer.run_end_events[0]
    assert end.n_stages == 4
    assert end.global_steps == 10
    assert end.transition_count == 3


def test_stage_metadata_mapping():
    """§51: callback global indices map to stages 0..3 with no skips/dupes."""
    observer = RecordingObserver()
    run_pipeline(FOUR_STAGE_CFG, observer)

    # Stage k owns global denoising indices [boundary_{k-1} .. boundary_k - 1]
    # (stage 0 starts at 0; the final stage runs to the schedule end).
    boundaries = (3, 5, 8)
    stage_starts = (0,) + boundaries
    for event in observer.step_events:
        expected_stage = max(
            i for i, start in enumerate(stage_starts) if event.global_schedule_index >= start
        )
        assert event.stage_index == expected_stage, (
            f"global {event.global_schedule_index} ran in stage {event.stage_index}, "
            f"expected {expected_stage}"
        )
        assert event.stage_local_step == event.global_schedule_index - stage_starts[expected_stage]

    # No skipped or duplicated denoising index.
    indices = observer.global_indices
    assert indices == sorted(set(indices)) == list(range(10))

    # Stage scales and geometry come from the actual stage slice.
    scales = (0.25, 0.5, 0.75, 1.0)
    stage_counts = {0: 3, 1: 2, 2: 3, 3: 2}
    for event in observer.step_events:
        assert event.stage_scale == scales[event.stage_index]
        assert (event.stage_h, event.stage_w) == (
            round(8 * scales[event.stage_index]), round(8 * scales[event.stage_index]),
        )
        assert event.full_h == 8 and event.full_w == 8 and event.full_t == 2
    for stage_idx, count in stage_counts.items():
        assert sum(1 for e in observer.step_events if e.stage_index == stage_idx) == count

    # Transition events line up with the resolved boundaries.
    transitions = observer.transition_events
    assert [(t.from_stage, t.to_stage) for t in transitions] == [(0, 1), (1, 2), (2, 3)]
    assert [t.global_schedule_index for t in transitions] == [3, 5, 8]
    assert [t.transition_index for t in transitions] == [0, 1, 2]
    assert [t.from_scale for t in transitions] == [0.25, 0.5, 0.75]
    assert [t.to_scale for t in transitions] == [0.5, 0.75, 1.0]
    for t in transitions:
        assert t.scale_ratio == pytest.approx(t.to_scale / t.from_scale)
        assert t.source_h < t.target_h and t.source_w < t.target_w


def test_actual_vs_original_sigma_at_boundaries():
    """§52: after each transition actual_sigma is the aligned (patched) working
    sigma; original_sigma is the untouched input schedule entry. They differ
    exactly by the kappa alignment."""
    observer = RecordingObserver()
    run_pipeline(FOUR_STAGE_CFG, observer)

    orig = [float(s) for s in SIGMAS_10]
    scales = (0.25, 0.5, 0.75, 1.0)
    boundaries = (3, 5, 8)

    # First callback of each post-transition stage sits at its boundary.
    first_after = {}
    for event in observer.step_events:
        if event.global_schedule_index in boundaries and event.global_schedule_index not in first_after:
            first_after[event.global_schedule_index] = event
    assert set(first_after) == set(boundaries)

    for boundary_idx, boundary in enumerate(boundaries):
        event = first_after[boundary]
        ratio = scales[boundary_idx + 1] / scales[boundary_idx]
        kappa, aligned = aligned_sigma(orig[boundary], ratio)

        assert event.actual_sigma == pytest.approx(aligned), (
            f"actual_sigma at boundary {boundary} must be the patched working sigma"
        )
        assert event.original_sigma == pytest.approx(orig[boundary])
        assert event.original_sigma_next == pytest.approx(orig[boundary + 1])
        assert event.actual_sigma != pytest.approx(event.original_sigma), (
            "kappa alignment must move the boundary sigma for this schedule"
        )

        # The transition event agrees: before/after alignment straddle it.
        transition = observer.transition_events[boundary_idx]
        assert transition.global_schedule_index == boundary
        assert transition.sigma_before_alignment == pytest.approx(orig[boundary])
        assert transition.sigma_after_alignment == pytest.approx(aligned)
        assert transition.kappa == pytest.approx(kappa)

    # Interior callbacks of the untouched first stage carry schedule values.
    for event in observer.step_events:
        if event.stage_index == 0:
            assert event.actual_sigma == pytest.approx(event.original_sigma)
            assert event.actual_sigma_next == pytest.approx(event.original_sigma_next)


def test_coincident_transitions_zero_step_stage():
    """§53: delta_custom resolving to (1,1) on the 8x8 latent calibration.

    Stage 0 steps -> transition A -> NO step in the zero-step intermediate
    stage -> transition B (whose before-alignment value is what transition A
    patched) -> final stage callbacks.
    """
    sigmas = torch.linspace(1.0, 0.0, 11)
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 1.0),
        transition_steps=(3, 5),  # ignored by delta_custom
        transition_mode="delta_custom",
        delta=0.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=8,
        full_latent_w=8,
    )
    observer = RecordingObserver()
    guider, _out, _denoised = run_pipeline(
        cfg, observer, sigmas=sigmas, latent=make_latent(t=2, h=8, w=8),
    )

    # The fake guider runs the zero-step intermediate stage as a single-entry
    # sample call: no callbacks fire inside it.
    assert len(guider.sigma_calls) == 3
    assert len(guider.sigma_calls[1]) == 1

    # Stage 0: global 0 only (boundary 1); final stage: global 1..9.
    assert observer.global_indices == [0] + list(range(1, 10))
    assert observer.step_events[0].stage_index == 0
    assert all(e.stage_index == 2 for e in observer.step_events[1:])
    assert not [e for e in observer.step_events if e.stage_index == 1]

    # Exactly two transition events, both at the coincident boundary.
    transitions = observer.transition_events
    assert len(transitions) == 2
    assert all(t.global_schedule_index == 1 for t in transitions)
    assert [(t.from_stage, t.to_stage) for t in transitions] == [(0, 1), (1, 2)]

    # Transition B reads the coordinate transition A patched.
    orig = [float(s) for s in sigmas]
    ratio_a = 0.5 / 0.25
    ratio_b = 1.0 / 0.5
    _kappa_a, new_q_a = aligned_sigma(orig[1], ratio_a)
    _kappa_b, new_q_b = aligned_sigma(new_q_a, ratio_b)

    assert transitions[0].sigma_before_alignment == pytest.approx(orig[1])
    assert transitions[0].sigma_after_alignment == pytest.approx(new_q_a)
    assert transitions[1].sigma_before_alignment == pytest.approx(new_q_a), (
        "transition B must read transition A's patched boundary value"
    )
    assert transitions[1].sigma_after_alignment == pytest.approx(new_q_b)
    assert transitions[1].sigma_after_alignment != pytest.approx(transitions[1].sigma_before_alignment)

    # The final stage re-enters at the double-aligned coordinate.
    assert observer.step_events[1].actual_sigma == pytest.approx(new_q_b)
    assert observer.step_events[1].original_sigma == pytest.approx(orig[1])


def test_observer_is_behaviorally_inert():
    """§54: the same deterministic fake generation with observer=None vs a
    RecordingObserver -> identical outputs, identical denoised, identical
    guider sigma calls, identical transition schedule."""

    def run(observer):
        guider = RecordingGuider()
        out, denoised = run_speed_pipeline(
            RecordingNoise(), guider, SIGMAS_10, make_latent(), FOUR_STAGE_CFG,
            sampler=type("S", (), {"name": "euler"})(),
            disable_pbar=True,
            observer=observer,
        )
        return guider, out, denoised

    guider_none, out_none, denoised_none = run(None)
    observer = RecordingObserver()
    guider_obs, out_obs, denoised_obs = run(observer)

    samples_none = list(out_none["samples"].unbind())
    samples_obs = list(out_obs["samples"].unbind())
    assert torch.equal(samples_none[0], samples_obs[0])
    assert torch.equal(samples_none[1], samples_obs[1])
    assert torch.equal(denoised_none["samples"], denoised_obs["samples"])
    assert guider_none.sigma_calls == guider_obs.sigma_calls

    # Transition schedule identical: 4 stage calls, canonical global slices,
    # total denoising steps == len(sigmas) - 1.
    orig = [float(s) for s in SIGMAS_10]
    assert len(guider_obs.sigma_calls) == 4
    assert guider_obs.sigma_calls[0] == orig[0:4]
    total_steps = sum(len(call) - 1 for call in guider_obs.sigma_calls)
    assert total_steps == len(SIGMAS_10) - 1

    # The observing run saw the full event stream.
    assert observer.global_indices == list(range(10))
    assert len(observer.transition_events) == 3


def test_step_event_callback_indices_are_global():
    """callback_index counts denoising intervals across the whole run."""
    observer = RecordingObserver()
    run_pipeline(FOUR_STAGE_CFG, observer)
    assert [e.callback_index for e in observer.step_events] == list(range(10))
