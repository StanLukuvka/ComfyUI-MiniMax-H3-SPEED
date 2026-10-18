"""SAMPLER pass-through, stage, progress, noise-policy, and I2V runtime contracts."""

import pytest
import torch

from conftest import (
    RecordingEchoGuider,
    StubSampler,
    make_fake_noise,
    make_latent,
    make_nested,
    make_recording_guider,
)
from speed_scripts import h3_runtime
from speed_scripts.planning import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])

#: Explicit stage ladders for the 2/3/4-stage tests: production Automatic
#: scale ladders with unique, strictly increasing global boundaries that tile
#: the 10-interval schedule (5 + 5, 3 + 2 + 5, 2 + 2 + 3 + 3).
LADDER_BOUNDARIES = {2: (5,), 3: (3, 5), 4: (2, 4, 7)}


class SeededRandomNoise:
    """Seeded random noise for deterministic non-zero test runs."""

    def __init__(self, seed=42):
        self.seed = seed

    def generate_noise(self, latent):
        video, audio = list(latent["samples"].unbind())
        generator = torch.Generator().manual_seed(self.seed)
        return make_nested(
            torch.randn(video.shape, generator=generator),
            torch.randn(audio.shape, generator=generator),
        )


def _cfg(**kwargs):
    kwargs.setdefault("scales", (.33, .66, 1.0))
    kwargs.setdefault("transition_steps", (3, 5))
    return SpeedConfig(transition_mode="explicit", **kwargs)


def _explicit_ladder_cfg(stages):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
    )


def _automatic_calibrated_cfg(stages):
    """Automatic config that produces coincident boundaries on this test schedule."""
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=tuple(range(1, len(STAGES_TO_SCALES[stages]))),
        transition_mode="delta_custom",
        delta=.005,
        noise_amplitude=12.105,
        noise_decay_exponent=.773,
    )


def _run(cfg, guider, sampler=None, *, noise=None, **kwargs):
    return run_speed_pipeline(
        noise if noise is not None else SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler=sampler if sampler is not None else StubSampler(),
        disable_pbar=True, **kwargs,
    )


def _assert_full_res_nested(latent):
    """Assert full-resolution nested video and audio output."""
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


# ---------------------------------------------------------------------------
# SAMPLER input contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3, 4))
def test_every_stage_receives_the_same_sampler_object_and_seed(stages):
    sampler = StubSampler("heun")
    guider = RecordingEchoGuider()
    _run(_explicit_ladder_cfg(stages), guider, sampler)
    assert len(guider.samplers) == stages
    assert all(seen is sampler for seen in guider.samplers)
    assert guider.seeds == [42] * stages


# ---------------------------------------------------------------------------
# Stage behavior
# ---------------------------------------------------------------------------

def test_stage_geometry_follows_the_scale_ladder():
    """Stage cond geometry follows the configured scale ladder and each
    stage consumes noise at its own grid: coarse 3x3, expanded 5x5, final 8x8."""
    stage_shapes = []
    guider = make_recording_guider(stage_shapes=stage_shapes)
    noise_guider = RecordingEchoGuider()
    _run(_cfg(), guider, noise=make_fake_noise())
    _run(_cfg(), noise_guider, noise=make_fake_noise())
    # round(8 * .33) = 3, round(8 * .66) = 5, final stage at full 8; the
    # noise video keeps its leading batch/channel dims (1, 1, t, h, w).
    assert stage_shapes == [(3, 3), (5, 5), (8, 8)]
    assert noise_guider.noise_shapes == [(1, 1, 2, 3, 3), (1, 1, 2, 5, 5), (1, 1, 2, 8, 8)]


@pytest.mark.parametrize("stages", (2, 3, 4))
def test_stage_ladder_completes_at_full_resolution(stages):
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(_explicit_ladder_cfg(stages), guider)

    assert len(guider.sigma_calls) == stages
    assert len(guider.noise_shapes) == stages
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


def test_run_is_deterministic_with_non_trivial_output():
    """Same seed and same fake model: two runs produce the same final video."""
    out, _ = _run(_explicit_ladder_cfg(3), RecordingEchoGuider(video_offset=.5))
    again, _ = _run(_explicit_ladder_cfg(3), RecordingEchoGuider(video_offset=.5))
    video, audio = out["samples"].unbind()
    in_video, in_audio = make_latent()["samples"].unbind()
    assert video.shape == in_video.shape
    assert audio.shape == in_audio.shape
    assert torch.equal(video, again["samples"].unbind()[0])
    assert video.abs().sum() > 0


def test_stage_failure_stops_later_stages():
    events = []

    class ExplodingStageGuider(RecordingEchoGuider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            events.append("sample")
            if len(events) == 2:
                raise RuntimeError("stage exploded")
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    with pytest.raises(RuntimeError, match="stage exploded"):
        _run(_explicit_ladder_cfg(3), ExplodingStageGuider())
    assert events == ["sample", "sample"]  # the final stage never ran


def test_second_generation_after_failure_starts_clean():
    """The same guider starts the next generation with clean state."""
    class ExplodingStageGuider(RecordingEchoGuider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.remaining_stage_failures = 0

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if self.remaining_stage_failures > 0:
                self.remaining_stage_failures -= 1
                raise RuntimeError("first generation exploded")
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    cfg = _explicit_ladder_cfg(3)
    guider_a = ExplodingStageGuider()
    guider_a.remaining_stage_failures = 1
    with pytest.raises(RuntimeError, match="first generation exploded"):
        _run(cfg, guider_a)

    out_second, _ = _run(cfg, guider_a)
    out_control, _ = _run(cfg, ExplodingStageGuider())

    video_second, audio_second = out_second["samples"].unbind()
    video_control, audio_control = out_control["samples"].unbind()
    assert torch.equal(video_second, video_control)
    assert torch.equal(audio_second, audio_control)
    _assert_full_res_nested(out_second)


# ---------------------------------------------------------------------------
# Noise policies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_policy", ("direct_coarse", "coupled_full_grid"))
def test_noise_policy_smoke(noise_policy):
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(_cfg(noise_policy=noise_policy), guider)
    assert len(guider.noise_shapes) == 3
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


def test_coupled_full_grid_transforms_full_noise_once(monkeypatch):
    original = h3_runtime.dct_temporal
    full_grid_calls = []

    def recording_dct_temporal(value):
        if tuple(value.shape[-2:]) == (8, 8):
            full_grid_calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(h3_runtime, "dct_temporal", recording_dct_temporal)
    _run(_cfg(noise_policy="coupled_full_grid"), RecordingEchoGuider(video_offset=.5))
    assert len(full_grid_calls) == 1


# ---------------------------------------------------------------------------
# Coincident boundaries
# ---------------------------------------------------------------------------

def test_coincident_four_stage_ladder_completes_with_zero_step_stages():
    guider = RecordingEchoGuider()
    out, _ = _run(_automatic_calibrated_cfg(4), guider)

    assert len(guider.sigma_calls) == 4
    for call in guider.sigma_calls[1:-1]:
        assert len(call) == 1
    _assert_full_res_nested(out)


# ---------------------------------------------------------------------------
# I2V keyframe lifecycle
# ---------------------------------------------------------------------------

def test_restore_runs_before_final_stage_and_again_in_cleanup(monkeypatch):
    order = []
    monkeypatch.setattr(
        h3_runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    _run(_cfg(), make_recording_guider(), noise=make_fake_noise())
    assert order == ["apply_final", "apply_final"]


def test_restore_is_retried_in_cleanup_when_it_fails(monkeypatch):
    order = []

    def exploding_final(self):
        order.append("apply_final")
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(h3_runtime.LatentWalker, "apply_final", exploding_final)
    with pytest.raises(RuntimeError, match="restore exploded"):
        _run(_cfg(), make_recording_guider(), noise=make_fake_noise())
    assert order == ["apply_final", "apply_final"]


def _i2v_guider():
    """RecordingEchoGuider with I2V-shaped conditioning attached.

    Returns (guider, keyframes, refs, original_conds).
    """
    guider = RecordingEchoGuider(video_offset=.5)
    keyframes = [{"latent": torch.zeros(1, 1, 2, 8, 8)} for _ in range(2)]
    refs = [{"latent": torch.zeros(1, 1, 2, 8, 8)} for _ in range(2)]
    guider.original_conds = {
        "positive": [{
            "minimax_keyframes": keyframes,
            "minimax_refs": refs,
        }],
        "negative": [],
    }
    return guider, keyframes, refs, guider.original_conds


def test_i2v_smoke_and_pristine_restore_on_success():
    guider, keyframes, refs, original_conds = _i2v_guider()
    out, denoised = _run(_explicit_ladder_cfg(3), guider)

    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    for kf in keyframes:
        assert tuple(kf["latent"].shape[-2:]) == (8, 8)
    for ref in refs:
        assert tuple(ref["latent"].shape[-2:]) == (8, 8)
    assert guider.original_conds is original_conds
    cond = original_conds["positive"][0]
    assert cond["minimax_keyframes"] is keyframes
    assert cond["minimax_refs"] is refs
