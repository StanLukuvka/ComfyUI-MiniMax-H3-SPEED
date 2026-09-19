"""Cross-sampler contracts for the native stateless samplers.

Each sampler uses the same SPEED stage schedule, progress timeline, and
resolution transitions. RES has separate stateful tests.
"""

import pytest
import torch

from conftest import (
    LADDER_BOUNDARIES,
    RecordingEchoGuider,
    SeededRandomNoise,
    make_latent,
)
from speed_scripts.planning import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    create_speed_sampler_handle,
)

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])

#: Stateless samplers tested against the Euler schedule baseline.
NEW_SAMPLERS = ("heun", "dpm_2", "exp_heun_2_x0")


def _nested(video, audio):
    return type(
        "Nested",
        (),
        {"is_nested": True, "unbind": lambda self: [video, audio]},
    )()


def _explicit_ladder_cfg(stages):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
    )


def _automatic_calibrated_cfg(stages):
    """Automatic config that produces coincident boundaries on this schedule."""
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=tuple(range(1, len(STAGES_TO_SCALES[stages]))),
        transition_mode="delta_custom",
        delta=.005,
        noise_amplitude=12.105,
        noise_decay_exponent=.773,
    )


def _run(sampler, cfg, guider, **kwargs):
    return run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name=sampler, disable_pbar=True, **kwargs,
    )


def _assert_full_res_nested(latent):
    """Assert full-resolution nested video and audio output."""
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


# ---------------------------------------------------------------------------
# 2/3/4-stage completion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
@pytest.mark.parametrize("stages", (2, 3, 4))
def test_stage_ladder_completes_at_full_resolution(sampler, stages):
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(sampler, _explicit_ladder_cfg(stages), guider)

    assert len(guider.sigma_calls) == stages
    assert len(guider.noise_shapes) == stages
    # Both outputs finish at full resolution with video and audio intact.
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


# ---------------------------------------------------------------------------
# Shared stage scheduling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
def test_stage_slices_match_the_euler_baseline(sampler):
    cfg = _explicit_ladder_cfg(3)
    baseline = RecordingEchoGuider()
    _run("euler", cfg, baseline)
    guider = RecordingEchoGuider()
    _run(sampler, cfg, guider)
    assert guider.sigma_calls == baseline.sigma_calls


@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
def test_every_stage_samples_through_the_selected_native_sampler(sampler):
    guider = RecordingEchoGuider()
    _run(sampler, _explicit_ladder_cfg(3), guider)
    # Every stage receives the selected native sampler.
    assert guider.samplers == [("sampler", sampler)] * 3


# ---------------------------------------------------------------------------
# Coincident boundaries and zero-step stages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
@pytest.mark.parametrize("stages", (3, 4))
def test_coincident_boundary_ladders_complete_with_zero_step_stages(sampler, stages):
    cfg = _automatic_calibrated_cfg(stages)
    guider = RecordingEchoGuider()
    out, _ = _run(sampler, cfg, guider)

    assert len(guider.sigma_calls) == stages
    # Middle stages may contain one sigma and zero denoising intervals.
    for call in guider.sigma_calls[1:-1]:
        assert len(call) == 1
    _assert_full_res_nested(out)


# ---------------------------------------------------------------------------
# Public progress
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
def test_callback_count_equals_global_denoising_intervals(sampler):
    seen = []
    guider = RecordingEchoGuider(video_offset=.5)
    _run(
        sampler, _explicit_ladder_cfg(3), guider,
        preview_callback=lambda step, x0, x, total: seen.append((step, total)),
    )
    # Ten denoising intervals produce ten global callbacks.
    assert seen == [(i, 10) for i in range(10)]


# ---------------------------------------------------------------------------
# Exp Heun 2 X0 reproducibility
# ---------------------------------------------------------------------------

def test_exp_heun_2_x0_repeat_run_is_reproducible():
    cfg = _explicit_ladder_cfg(3)
    out, _ = _run("exp_heun_2_x0", cfg, RecordingEchoGuider(video_offset=.5))
    out_again, _ = _run(
        "exp_heun_2_x0", cfg, RecordingEchoGuider(video_offset=.5)
    )
    video, _ = out["samples"].unbind()
    again_video, _ = out_again["samples"].unbind()
    assert torch.equal(video, again_video)
    # Confirm the output contains non-zero signal.
    assert video.abs().sum() > 0


def test_stochastic_exp_heun_variant_is_rejected_fail_closed():
    with pytest.raises(ValueError) as excinfo:
        create_speed_sampler_handle("exp_heun_2_x0_sde")
    message = str(excinfo.value)
    assert "exp_heun_2_x0_sde" in message
    for supported in STATELESS_SPEED_SAMPLERS:
        assert supported in message
