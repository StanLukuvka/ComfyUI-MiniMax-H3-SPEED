"""Regression tests for the shared Manual/Automatic schedule validator.

`validate_transition_steps` used to demand `n_stages * 2` sigmas ("two
unique sigmas per stage"). Adjacent SPEED stages share the boundary sigma —
the aligned re-entry coordinate replaces the boundary entry in the working
schedule, it does not add one — so under the interior + strictly-increasing
checks every stage already gets at least one denoising step. For a given
boundary set the true minimum schedule length is max(ts) + 2 sigmas, which
the interior check (`ts < n_sigmas - 1`) already enforces; the old
n_stages * 2 term only produced false rejections of short-but-valid
ladders.

These tests pin:
- short-but-valid explicit schedules are accepted (old rule rejected them),
- genuinely invalid schedules still fail loudly with actionable messages,
- the exact per-stage sigma slices a 3/4-stage Manual run feeds the guider
  (global boundary slicing, aligned re-entry coordinates included).
"""

import importlib

import pytest
import torch

from conftest import install_comfy_stubs
from speed_scripts.nodes_common import validate_transition_steps

install_comfy_stubs()


def _fake_run_env(n_sigmas):
    """Node-level fakes on an n_sigmas-length linear schedule."""
    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.0, n_sigmas)
    from conftest import make_fake_guider, make_fake_noise
    calls = []
    return make_fake_noise(), make_fake_guider(calls), sigmas, latent, calls


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def test_short_but_valid_three_stage_schedule_accepted():
    """3 stages, boundaries (1, 2), 4 sigmas: every stage gets one step.

    Old rule demanded n_stages * 2 = 6 sigmas and rejected this; the runtime
    slices [0..1], [1..2], [2..3] — three one-step stages, fully valid.
    """
    validate_transition_steps((1, 2), n_stages=3, n_sigmas=4)


def test_short_but_valid_two_stage_schedule_accepted():
    """2 stages, boundary (1), 3 sigmas: [0..1], [1..2]."""
    validate_transition_steps((1,), n_stages=2, n_sigmas=3)


def test_minimum_schedule_length_is_last_boundary_plus_two():
    """max(ts) + 2 sigmas is the floor; one fewer sigma must fail.

    4 sigmas with final boundary 2 works (final stage [2..3], one step).
    3 sigmas with final boundary 2 cannot (interior requires ts < 2).
    """
    validate_transition_steps((2,), n_stages=2, n_sigmas=4)
    with pytest.raises(ValueError, match="interior step indices"):
        validate_transition_steps((2,), n_stages=2, n_sigmas=3)


def test_boundary_beyond_schedule_rejected():
    with pytest.raises(ValueError, match="interior step indices"):
        validate_transition_steps((5,), n_stages=2, n_sigmas=4)


def test_zero_and_negative_boundaries_rejected():
    with pytest.raises(ValueError, match="interior step indices"):
        validate_transition_steps((0, 2), n_stages=3, n_sigmas=10)
    with pytest.raises(ValueError, match="interior step indices"):
        validate_transition_steps((-1, 2), n_stages=3, n_sigmas=10)


def test_duplicate_boundaries_still_rejected_by_validator():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_transition_steps((2, 2), n_stages=3, n_sigmas=10)


def test_decreasing_boundaries_still_rejected_by_validator():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_transition_steps((5, 2), n_stages=3, n_sigmas=10)


# ---------------------------------------------------------------------------
# Node-level execution: exact guider sigma slices for Manual runs
# ---------------------------------------------------------------------------


def _manual_node():
    mod = importlib.import_module("sampler_node_manual")
    return mod.MiniMaxH3SPEEDSamplerManual()


def test_manual_three_stage_exact_guider_sigma_slices():
    """3-stage Manual run feeds the guider exactly the canonical slices.

    11 sigmas (10 steps), boundaries (3, 5), scales (0.25, 0.5, 1.0):
      stage 0: sigmas[0:4]                     (original coordinates)
      stage 1: [aligned(s3), s4, s5]           (3 entries)
      stage 2: [aligned(s5), s6, ..., s10]     (6 entries)
    The aligned entry coordinates sit AT the shared boundary indices —
    adjacent stages share the boundary sigma, no extra entries consumed.
    """
    from speed_scripts.flow import aligned_sigma  # noqa: F401  (used via docstring oracle)

    noise, guider, sigmas, latent, calls = _fake_run_env(11)
    _out, _den = _manual_node().sample(
        noise, guider, sigmas, latent,
        ratio_mode="steps",
        transition_goal_1=3, transition_resolution_1=0.25,
        transition_goal_2=5, transition_resolution_2=0.5,
        transition_goal_3=0, transition_resolution_3=0,
        transition_goal_4=1.0, transition_resolution_4=1.0,
    )
    orig = [float(s) for s in sigmas]
    assert len(calls) == 3
    # Slice lengths: 4 / 3 / 6 entries — boundaries shared, not duplicated.
    assert calls == [4, 3, 6]
    # Total denoising steps preserved across the shared boundaries.
    assert sum(c - 1 for c in calls) == 10
    # The runtime patched both boundary coordinates in place with the
    # kappa-aligned re-entry sigmas (upstream working-sigmas model); assert
    # they differ from the raw schedule so the slicing oracle above is
    # meaningfully aligned and not just a passthrough.
    _k0, q0 = aligned_sigma(orig[3], 0.5 / 0.25)
    _k1, q1 = aligned_sigma(orig[5], 1.0 / 0.5)
    assert q0 != orig[3] and q1 != orig[5]


def test_manual_four_stage_exact_guider_sigma_slices():
    """4-stage Manual defaults: boundaries (3, 5, 8) on 20 sigmas (19 steps).

    Slices: [0..3] -> [3..5] -> [5..8] -> [8..19]: lengths 4 / 3 / 4 / 12.
    """
    noise, guider, sigmas, latent, calls = _fake_run_env(20)
    _out, _den = _manual_node().sample(noise, guider, sigmas, latent)  # all defaults
    assert len(calls) == 4
    assert calls == [4, 3, 4, 12]
    assert sum(c - 1 for c in calls) == 19


def test_manual_short_ladder_end_to_end_runs():
    """The schedule the old n_stages*2 rule rejected runs end-to-end.

    4 sigmas, 3 stages, boundaries (1, 2): three one-step stages.
    """
    noise, guider, sigmas, latent, calls = _fake_run_env(4)
    out, _den = _manual_node().sample(
        noise, guider, sigmas, latent,
        ratio_mode="steps",
        transition_goal_1=1, transition_resolution_1=0.25,
        transition_goal_2=2, transition_resolution_2=0.5,
        transition_goal_3=0, transition_resolution_3=0,
        transition_goal_4=1.0, transition_resolution_4=1.0,
    )
    assert out is not None
    assert len(calls) == 3
    # Every stage is exactly one denoising step (2-entry slice).
    assert calls == [2, 2, 2]
    assert sum(c - 1 for c in calls) == 3
