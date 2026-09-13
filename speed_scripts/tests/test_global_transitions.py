"""Execution-level contracts for global SPEED transition scheduling."""

import pytest
import torch

from conftest import make_fake_noise, make_latent, make_recording_guider
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import resolve_transition_steps, run_speed_pipeline


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def _expected_schedules(sigmas, boundaries, scales):
    working = [float(s) for s in sigmas]
    starts = [0, *boundaries]
    ends = [*boundaries, len(working) - 1]
    schedules = []
    for stage, (start, end) in enumerate(zip(starts, ends)):
        schedules.append(list(working[start:end + 1]))
        if stage < len(boundaries):
            ratio = scales[stage + 1] / scales[stage]
            _, working[end] = aligned_sigma(working[end], ratio)
    return schedules


def _run(config, sigmas=SIGMAS, latent=None, preview_callback=None):
    calls = []
    guider = make_recording_guider(sigma_calls=calls)
    output = run_speed_pipeline(
        make_fake_noise(),
        guider,
        sigmas,
        latent or make_latent(),
        config,
        sampler_override=object(),
        disable_pbar=True,
        preview_callback=preview_callback,
    )
    return calls, output


@pytest.mark.parametrize("noise_policy", ["direct_coarse", "coupled_full_grid"])
def test_explicit_four_stage_schedule_uses_global_boundaries(noise_policy):
    scales = (.25, .5, .75, 1.0)
    boundaries = (3, 5, 8)
    cfg = SpeedConfig(
        scales=scales,
        transition_steps=boundaries,
        transition_mode="explicit",
        noise_policy=noise_policy,
    )
    calls, _ = _run(cfg)
    expected = _expected_schedules(SIGMAS, boundaries, scales)

    assert len(calls) == 4
    for actual, wanted in zip(calls, expected):
        assert actual == pytest.approx(wanted)
    assert sum(len(call) - 1 for call in calls) == len(SIGMAS) - 1


def test_delta_custom_execution_matches_its_resolved_global_boundaries():
    sigmas = torch.linspace(1.0, 0.0, 21)
    scales = (.25, .5, 1.0)
    cfg = SpeedConfig(
        scales=scales,
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=44,
        full_latent_w=80,
    )
    boundaries = resolve_transition_steps(cfg, sigmas, H_full=44, W_full=80)
    calls, _ = _run(cfg, sigmas, make_latent(h=44, w=80))
    expected = _expected_schedules(sigmas, boundaries, scales)

    assert len(calls) == len(scales)
    for actual, wanted in zip(calls, expected):
        assert actual == pytest.approx(wanted)
    assert sum(len(call) - 1 for call in calls) == len(sigmas) - 1


def test_coincident_delta_boundaries_align_the_same_coordinate_twice():
    sigmas = torch.linspace(1.0, 0.0, 11)
    scales = (.25, .5, 1.0)
    cfg = SpeedConfig(
        scales=scales,
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=8,
        full_latent_w=8,
    )
    boundaries = resolve_transition_steps(cfg, sigmas, H_full=8, W_full=8)
    assert boundaries == (1, 1)

    calls, _ = _run(cfg, sigmas, make_latent(h=8, w=8))
    original = [float(s) for s in sigmas]
    _, once = aligned_sigma(original[1], 2.0)
    _, twice = aligned_sigma(once, 2.0)

    assert calls[0] == pytest.approx(original[:2])
    assert calls[1] == pytest.approx([once])  # zero denoising intervals
    assert calls[2][0] == pytest.approx(twice)
    assert calls[2][1:] == pytest.approx(original[2:])
    assert sum(len(call) - 1 for call in calls) == len(sigmas) - 1


def test_preview_callback_sees_one_continuous_global_timeline():
    seen = []

    def preview(step, x0, x, total_steps):
        seen.append((step, total_steps))

    cfg = SpeedConfig(
        scales=(.25, .5, .75, 1.0),
        transition_steps=(3, 5, 8),
        transition_mode="explicit",
    )
    _run(cfg, preview_callback=preview)
    assert seen == [(i, 10) for i in range(10)]


def test_runtime_rejects_boundary_at_schedule_end():
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(10,),
        transition_mode="explicit",
    )
    with pytest.raises(ValueError, match="inside the sigma schedule"):
        _run(cfg)
