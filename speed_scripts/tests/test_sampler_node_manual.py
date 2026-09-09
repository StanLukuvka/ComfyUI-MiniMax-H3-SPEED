"""Manual explicit-schedule node contracts."""

import importlib

import pytest
import torch

from conftest import make_fake_noise, make_latent, make_recording_guider
from speed_scripts.flow import aligned_sigma


def _run_manual(sigmas, **kwargs):
    mod = importlib.import_module("sampler_node_manual")
    sigma_calls, shapes = [], []
    output = mod.MiniMaxH3SPEEDSamplerManual().sample(
        make_fake_noise(),
        make_recording_guider(sigma_calls=sigma_calls, stage_shapes=shapes),
        sigmas,
        make_latent(h=8, w=8),
        **kwargs,
    )
    return output, sigma_calls, shapes


def test_manual_node_public_surface():
    cls = importlib.import_module("sampler_node_manual").MiniMaxH3SPEEDSamplerManual
    required = cls.INPUT_TYPES()["required"]
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert {"noise", "guider", "sigmas", "latent_image", "ratio_mode",
            "transition_goal_1", "transition_resolution_1",
            "transition_goal_4", "transition_resolution_4"} <= set(required)
    assert required["transition_resolution_1"][1]["default"] == 0.25
    assert required["transition_resolution_4"][1]["default"] == 1.0


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
        # Final goal is intentionally the normal unused default (>1).
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
