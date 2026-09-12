"""Cross-sampler execution tests for the PR A stateless samplers
(plan §9 Heun / DPM2 / Exp Heun 2 X0 blocks).

All stateless samplers share one runtime path — the run-scoped handle feeds
the same stage loop — so the checks run parameterized over ``heun``,
``dpm_2`` and ``exp_heun_2_x0``, with ``euler`` (the S2 regression anchor,
already pinned in ``test_sampler_support.py``) appearing only as the
differential baseline for stage scheduling and inside the
coincident-boundary completion sweep.

What these tests pin is what SPEED owes every sampler: each stage hands the
solver its complete interval set in one ``guider.sample`` call (so a
multi-evaluation solver's extra model evaluations stay inside one interval
and a SPEED transition lands only between completed intervals), public
progress stays at one callback per global denoising interval, coincident
boundaries and zero-step stages complete, and the deterministic Exp Heun 2
X0 path reproduces exactly. Solver math itself stays inside ComfyUI's
native sampler objects and is never re-implemented or inspected here.
"""

import pytest
import torch

from conftest import (
    LADDER_BOUNDARIES,
    RecordingEchoGuider,
    SeededRandomNoise,
    make_latent,
)
from speed_scripts.automatic_config import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    create_speed_sampler_handle,
)

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])

#: The three samplers this slice adds on top of the Euler regression anchor.
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
    """The Automatic node's delta_custom config for ``stages`` stages.

    With the baked calibration constants on the 10-interval schedule every
    transition quantizes onto schedule index 1, so the middle stages get a
    single-sigma schedule (zero denoising steps) — the legal
    coincident-boundary case the runtime must support.
    """
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=tuple(range(1, len(STAGES_TO_SCALES[stages]))),
        transition_mode="delta_custom",
        delta=.005,
        noise_amplitude=12.105,
        noise_decay_exponent=.773,
        full_latent_h=8,
        full_latent_w=8,
    )


def _run(sampler, cfg, guider, **kwargs):
    return run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name=sampler, disable_pbar=True, **kwargs,
    )


def _assert_full_res_nested(latent):
    """Both H3 streams survived: 5-dim video + 4-dim audio at full 8x8."""
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


# ---------------------------------------------------------------------------
# Parameterized 2/3/4-stage completion (plan §9 Heun block; DPM2 and
# Exp Heun 2 X0 inherit via "same execution checks as Heun")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
@pytest.mark.parametrize("stages", (2, 3, 4))
def test_stage_ladder_completes_at_full_resolution(sampler, stages):
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(sampler, _explicit_ladder_cfg(stages), guider)

    assert len(guider.sigma_calls) == stages
    assert len(guider.noise_shapes) == stages
    # Nested H3 video+audio output survives on both node outputs, and the
    # final geometry is full resolution.
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


# ---------------------------------------------------------------------------
# Stage scheduling is sampler-independent (plan §2: scheduler logic never
# branches on the sampler name). Whole-interval slices in one call per stage
# also mean a Heun second evaluation / DPM2 midpoint stays inside one
# sampler interval, and a SPEED transition can only land between completed
# intervals.
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
    # The factory-built native Comfy sampler object for the selected name
    # reached every stage call — never a silent Euler substitute.
    assert guider.samplers == [("sampler", sampler)] * 3


# ---------------------------------------------------------------------------
# Coincident boundaries + zero-step stages complete (plan §9 Heun block;
# euler included to anchor the sweep)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
@pytest.mark.parametrize("stages", (3, 4))
def test_coincident_boundary_ladders_complete_with_zero_step_stages(sampler, stages):
    cfg = _automatic_calibrated_cfg(stages)
    guider = RecordingEchoGuider()
    out, _ = _run(sampler, cfg, guider)

    assert len(guider.sigma_calls) == stages
    # Every middle stage quantized onto the same boundary coordinate: its
    # schedule is a single sigma (zero denoising steps) yet the stage still
    # ran, and the run finished at full resolution.
    for call in guider.sigma_calls[1:-1]:
        assert len(call) == 1
    _assert_full_res_nested(out)


# ---------------------------------------------------------------------------
# Public progress (plan §9: callback count equals global denoising
# intervals, not model evaluations — Heun's second evaluation and DPM2's
# midpoint evaluation must not surface as extra public progress steps)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sampler", NEW_SAMPLERS)
def test_callback_count_equals_global_denoising_intervals(sampler):
    seen = []
    guider = RecordingEchoGuider(video_offset=.5)
    _run(
        sampler, _explicit_ladder_cfg(3), guider,
        preview_callback=lambda step, x0, x, total: seen.append((step, total)),
    )
    # A (3, 5) ladder over a 10-interval schedule forwards exactly 10
    # callbacks on one continuous global timeline.
    assert seen == [(i, 10) for i in range(10)]


# ---------------------------------------------------------------------------
# Exp Heun 2 X0 (plan §9: deterministic repeat-run reproducibility; no
# stochastic path)
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
    # The comparison is meaningful: the run carried non-trivial signal.
    assert video.abs().sum() > 0


def test_stochastic_exp_heun_variant_is_rejected_fail_closed():
    with pytest.raises(ValueError) as excinfo:
        create_speed_sampler_handle("exp_heun_2_x0_sde")
    message = str(excinfo.value)
    assert "exp_heun_2_x0_sde" in message
    for supported in STATELESS_SPEED_SAMPLERS:
        assert supported in message
