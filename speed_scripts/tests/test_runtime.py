"""run_speed_pipeline: stage schedules, outputs, noise policies, previews, I2V, audio, image runs."""

import pytest
import torch

from conftest import (
    SIGMAS,
    AffineModel,
    EchoGuider,
    EchoNoise,
    OracleFlowGuider,
    SeededNoise,
    StubSampler,
    expected_stage_sigmas,
    image_latent,
    make_latent,
)
from speed_scripts import runtime
from speed_scripts import spectral
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import (
    aligned_sigma,
    clock_reindex_audio_state,
    reentry_noise,
    time_shift_sigma,
    to_internal_state,
)
from speed_scripts.planning import STAGES_TO_SCALES, resolve_transition_steps
from speed_scripts.runtime import run_speed_pipeline
from speed_scripts.spectral import lowpass_dct

#: Explicit boundaries that tile the 10-step schedule for the Automatic 2/3/4-stage ladders.
LADDER_BOUNDARIES = {2: (5,), 3: (3, 5), 4: (2, 4, 7)}


def _run(config, guider=None, *, noise=None, latent=None, sigmas=SIGMAS, **kwargs):
    kwargs.setdefault("sampler", StubSampler())
    return run_speed_pipeline(
        noise if noise is not None else SeededNoise(),
        guider if guider is not None else EchoGuider(),
        sigmas,
        latent if latent is not None else make_latent(),
        config,
        disable_pbar=True,
        **kwargs,
    )


def _cfg(**kwargs):
    kwargs.setdefault("scales", (.33, .66, 1.0))
    kwargs.setdefault("transition_steps", (3, 5))
    return SpeedConfig(transition_mode="explicit", **kwargs)


def _ladder_cfg(stages):
    return _cfg(scales=STAGES_TO_SCALES[stages], transition_steps=LADDER_BOUNDARIES[stages])


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


def _assert_full_res_nested(latent):
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


# ---------------------------------------------------------------------------
# Stage schedule on the global sigma timeline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_policy", ["direct_coarse", "coupled_full_grid"])
def test_explicit_four_stage_schedule_uses_global_boundaries(noise_policy):
    scales = (.25, .5, .75, 1.0)
    boundaries = (3, 5, 8)
    guider = EchoGuider()
    _run(_cfg(scales=scales, transition_steps=boundaries, noise_policy=noise_policy), guider,
         noise=EchoNoise())
    calls = guider.sigma_calls

    assert len(calls) == 4
    for actual, wanted in zip(calls, expected_stage_sigmas(SIGMAS, boundaries, scales)):
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
    )
    boundaries = resolve_transition_steps(cfg, sigmas, H_full=44, W_full=80)
    guider = EchoGuider()
    _run(cfg, guider, noise=EchoNoise(), latent=make_latent(h=44, w=80), sigmas=sigmas)
    calls = guider.sigma_calls

    assert len(calls) == len(scales)
    for actual, wanted in zip(calls, expected_stage_sigmas(sigmas, boundaries, scales)):
        assert actual == pytest.approx(wanted)
    assert sum(len(call) - 1 for call in calls) == len(sigmas) - 1


def test_coincident_delta_boundaries_align_the_same_coordinate_twice():
    sigmas = torch.linspace(1.0, 0.0, 11)
    cfg = SpeedConfig(
        scales=(.25, .5, 1.0),
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
    )
    assert resolve_transition_steps(cfg, sigmas, H_full=8, W_full=8) == (1, 1)

    guider = EchoGuider()
    _run(cfg, guider, noise=EchoNoise(), sigmas=sigmas)
    calls = guider.sigma_calls
    original = [float(s) for s in sigmas]
    _, once = aligned_sigma(original[1], 2.0)
    _, twice = aligned_sigma(once, 2.0)

    assert calls[0] == pytest.approx(original[:2])
    assert calls[1] == pytest.approx([once])  # zero denoising intervals
    assert calls[2][0] == pytest.approx(twice)
    assert calls[2][1:] == pytest.approx(original[2:])
    assert sum(len(call) - 1 for call in calls) == len(sigmas) - 1


def test_coincident_four_stage_ladder_completes_with_zero_step_stages():
    guider = EchoGuider()
    out, _ = _run(_automatic_calibrated_cfg(4), guider)

    assert len(guider.sigma_calls) == 4
    for call in guider.sigma_calls[1:-1]:
        assert len(call) == 1
    _assert_full_res_nested(out)


def test_runtime_rejects_boundary_at_schedule_end():
    with pytest.raises(ValueError, match="inside the sigma schedule"):
        _run(_cfg(scales=(.5, 1.0), transition_steps=(10,)), noise=EchoNoise())


# ---------------------------------------------------------------------------
# Stages, sampler, and outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3, 4))
def test_every_stage_receives_the_same_sampler_object_and_seed(stages):
    sampler = StubSampler("heun")
    guider = EchoGuider()
    _run(_ladder_cfg(stages), guider, sampler=sampler)
    assert len(guider.samplers) == stages
    assert all(seen is sampler for seen in guider.samplers)
    assert guider.seeds == [42] * stages


def test_stage_geometry_follows_the_scale_ladder():
    """Stage cond geometry follows the configured scale ladder and each
    stage consumes noise at its own grid: coarse 3x3, expanded 5x5, final 8x8."""
    guider = EchoGuider()
    _run(_cfg(), guider, noise=EchoNoise())
    # round(8 * .33) = 3, round(8 * .66) = 5, final stage at full 8; the
    # noise video keeps its leading batch/channel dims (1, 1, t, h, w).
    assert guider.latent_sizes == [(3, 3), (5, 5), (8, 8)]
    assert guider.noise_shapes == [(1, 1, 2, 3, 3), (1, 1, 2, 5, 5), (1, 1, 2, 8, 8)]


@pytest.mark.parametrize("stages", (2, 3, 4))
def test_stage_ladder_completes_at_full_resolution(stages):
    guider = EchoGuider(video_offset=.5)
    out, denoised = _run(_ladder_cfg(stages), guider)

    assert len(guider.sigma_calls) == stages
    assert len(guider.noise_shapes) == stages
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


def test_run_is_deterministic_with_non_trivial_output():
    """Same seed and same fake model: two runs produce the same final video."""
    out, _ = _run(_ladder_cfg(3), EchoGuider(video_offset=.5))
    again, _ = _run(_ladder_cfg(3), EchoGuider(video_offset=.5))
    video, audio = out["samples"].unbind()
    in_video, in_audio = make_latent()["samples"].unbind()
    assert video.shape == in_video.shape
    assert audio.shape == in_audio.shape
    assert torch.equal(video, again["samples"].unbind()[0])
    assert video.abs().sum() > 0


def test_denoised_output_preserves_video_and_audio_streams():
    """Both returned LATENTs keep the full nested video+audio structure."""
    guider = EchoGuider()
    out, denoised = _run(_cfg(), guider, noise=EchoNoise(), x0_output={})

    seen = guider.model_patcher.model.latents_out
    assert seen, "process_latent_out never ran (no x0 captured)"
    nested_x0 = seen[-1]
    assert getattr(nested_x0, "is_nested", False), (
        "process_latent_out received a bare video tensor — the nested "
        "video+audio x0 was unpacked before reaching the model"
    )
    video, audio = nested_x0.unbind()
    assert video.ndim == 5 and audio.ndim == 4, "x0 streams unpacked wrong"

    for name, latent_out in (("output", out), ("denoised", denoised)):
        samples = latent_out["samples"]
        assert getattr(samples, "is_nested", False), f"{name} lost the nested bundle"
        v, a = samples.unbind()
        assert v.ndim == 5, f"{name} video stream missing/mangled"
        assert a.ndim == 4, f"{name} audio stream missing/mangled"


class _ExplodingStageGuider(EchoGuider):
    """EchoGuider whose next ``failures`` stage calls raise."""

    def __init__(self, failures=0, fail_at_call=None, **kwargs):
        super().__init__(**kwargs)
        self.failures = failures
        self.fail_at_call = fail_at_call
        self.calls = 0

    def sample(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at_call or self.failures > 0:
            self.failures = max(0, self.failures - 1)
            raise RuntimeError("stage exploded")
        return super().sample(*args, **kwargs)


def test_stage_failure_stops_later_stages():
    guider = _ExplodingStageGuider(fail_at_call=2)
    with pytest.raises(RuntimeError, match="stage exploded"):
        _run(_ladder_cfg(3), guider)
    assert guider.calls == 2  # the final stage never ran


def test_second_generation_after_failure_starts_clean():
    """The same guider starts the next generation with clean state."""
    cfg = _ladder_cfg(3)
    guider = _ExplodingStageGuider(failures=1)
    with pytest.raises(RuntimeError, match="stage exploded"):
        _run(cfg, guider)

    out_second, _ = _run(cfg, guider)
    out_control, _ = _run(cfg, EchoGuider())

    video_second, audio_second = out_second["samples"].unbind()
    video_control, audio_control = out_control["samples"].unbind()
    assert torch.equal(video_second, video_control)
    assert torch.equal(audio_second, audio_control)
    _assert_full_res_nested(out_second)


# ---------------------------------------------------------------------------
# Preview callbacks
# ---------------------------------------------------------------------------

def _four_stage_cfg():
    return _cfg(scales=(.25, .5, .75, 1.0), transition_steps=(3, 5, 8))


def test_preview_callback_sees_one_continuous_global_timeline():
    seen = []

    def preview(step, x0, x, total_steps):
        seen.append((step, total_steps))

    _run(_four_stage_cfg(), noise=EchoNoise(), preview_callback=preview)
    assert seen == [(i, 10) for i in range(10)]


def test_preview_callback_failure_disables_future_preview_updates():
    seen = []

    def preview(step, x0, x, total_steps):
        seen.append(step)
        raise RuntimeError("preview exploded")

    _run(_four_stage_cfg(), noise=EchoNoise(), preview_callback=preview)
    assert seen == [0]


# ---------------------------------------------------------------------------
# Noise policies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("noise_policy", ("direct_coarse", "coupled_full_grid"))
def test_noise_policy_smoke(noise_policy):
    guider = EchoGuider(video_offset=.5)
    out, denoised = _run(_cfg(noise_policy=noise_policy), guider)
    assert len(guider.noise_shapes) == 3
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


@pytest.mark.parametrize("temporal_scales", [(), (.5, .75, 1.0)], ids=["spatial", "temporal"])
def test_coupled_full_grid_projects_full_noise_in_the_spectral_domain(temporal_scales):
    """Coupled stages project one full-resolution noise field into each stage grid.

    The expected re-entry noise is rebuilt from the same DCT operations used
    by the runtime and compared at both transitions.
    """
    noise = SeededNoise(7)
    full_noise_video, full_noise_audio = noise.generate_noise(make_latent()).unbind()
    video_offset = 0.37  # makes the stage output (and source block) non-trivial
    audio_scale = 4.0

    guider = EchoGuider(video_offset=video_offset)
    out, _ = _run(_cfg(noise_policy="coupled_full_grid", temporal_scales=temporal_scales),
                  guider, noise=noise)
    pubs = [video for video, _ in guider.noise]

    full_t = 2
    # Stage sizes use the same rounding as the runtime.
    thw = [
        (
            full_t if not temporal_scales else max(1, round(full_t * temporal_scales[i])),
            max(1, round(8 * s)),
            max(1, round(8 * s)),
        )
        for i, s in enumerate((.33, .66, 1.0))
    ]
    assert [p.shape[-3:] for p in pubs] == thw

    # Full-resolution DCT shared by every coupled stage.
    full_dct = spectral.dct2(spectral.dct_temporal(full_noise_video))

    # Stage 0 starts from the low-frequency block of the full noise field.
    s0_t, s0_h, s0_w = thw[0]
    stage_pubs = [
        spectral.idct2(spectral.dct2(full_noise_video[..., :s0_t, :, :])[..., :s0_h, :s0_w])
    ]
    for transition, (boundary, ratio) in enumerate(((3, .66 / .33), (5, 1.0 / .66))):
        rsp_q = float(SIGMAS[boundary])
        internal = to_internal_state(
            stage_pubs[-1] + video_offset, full_noise_audio, rsp_q, audio_scale
        )[0]
        kappa, new_q = aligned_sigma(rsp_q, ratio)
        (st, sh, sw), (t, h, w) = thw[transition], thw[transition + 1]
        target_dct = full_dct[..., :t, :h, :w] * rsp_q
        target_dct[..., :st, :sh, :sw] = spectral.dct2(spectral.dct_temporal(internal))
        expanded = spectral.idct_temporal(spectral.idct2(target_dct))
        expected = reentry_noise(expanded * kappa, new_q)
        assert torch.allclose(pubs[transition + 1], expected, atol=1e-4), (
            f"transition {transition + 1} re-entry noise is not the spectral projection "
            "of the full-grid noise (pixel-space cropping drift?)"
        )
        stage_pubs.append(pubs[transition + 1])

    final_video, _ = out["samples"].unbind()
    assert tuple(final_video.shape[-3:]) == (full_t, 8, 8)


def test_coupled_full_grid_transforms_full_noise_once(monkeypatch):
    original = runtime.dct_temporal
    full_grid_calls = []

    def recording_dct_temporal(value):
        if tuple(value.shape[-2:]) == (8, 8):
            full_grid_calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(runtime, "dct_temporal", recording_dct_temporal)
    _run(_cfg(noise_policy="coupled_full_grid"), EchoGuider(video_offset=.5))
    assert len(full_grid_calls) == 1


# ---------------------------------------------------------------------------
# H3 audio across a transition
# ---------------------------------------------------------------------------

def test_audio_transition_oracle_clock_reindex_bridge():
    """Verify clock_reindex audio re-entry against the flow math."""
    sigmas = torch.tensor([1.0, .8, .6, .4, .2, 0.0])
    cfg = _cfg(scales=(.5, 1.0), transition_steps=(2,), audio_policy="clock_reindex")
    audio_out = 2.5  # non-zero constant the guider "denoises" to

    guider = EchoGuider(video_offset=.5, audio_value=audio_out)
    x0_output = {}
    out, _ = _run(cfg, guider, noise=EchoNoise(), sigmas=sigmas, x0_output=x0_output)

    # Stage-1 re-entry audio carries non-zero state.
    assert len(guider.noise) == 2
    _, reentry_audio = guider.noise[1]
    assert torch.count_nonzero(reentry_audio) > 0, "oracle audio collapsed to zero"

    video_shift, audio_shift = 12.0, 3.0
    audio_scale = video_shift / audio_shift  # 4.0 — as resolve_sigma_shifts computes it
    rsp_q = float(sigmas[2])  # stage-0 boundary (global index 2)
    kappa, new_q = aligned_sigma(rsp_q, 1.0 / .5)

    # The fixed stage-0 audio output in the internal representation.
    stage0_pub_audio = torch.full_like(reentry_audio, audio_out)
    internal_audio = to_internal_state(
        torch.zeros_like(stage0_pub_audio), stage0_pub_audio, rsp_q, audio_scale
    )[1]
    assert torch.count_nonzero(internal_audio) > 0

    # x0 audio stays in model-output space.
    _, x0_audio = x0_output["x0"].unbind()
    assert torch.count_nonzero(x0_audio) > 0, "x0 audio must be non-zero"

    transitioned_audio = clock_reindex_audio_state(
        internal_audio,
        x0_audio,
        rsp_q,
        new_q,
        time_shift_sigma(rsp_q, video_shift, audio_shift),
        time_shift_sigma(new_q, video_shift, audio_shift),
        audio_scale,
    )
    expected_reentry_audio = reentry_noise(transitioned_audio, new_q)
    assert torch.allclose(reentry_audio, expected_reentry_audio, atol=1e-5), (
        "stage-1 re-entry audio does not match the independent clock_reindex "
        "oracle (audio_scale=4.0)"
    )

    # Final output keeps the non-zero audio stream.
    _, final_audio = out["samples"].unbind()
    assert torch.count_nonzero(final_audio) > 0
    assert final_audio.shape == reentry_audio.shape


# ---------------------------------------------------------------------------
# I2V keyframe lifecycle
# ---------------------------------------------------------------------------

def test_restore_runs_before_final_stage_and_again_in_cleanup(monkeypatch):
    order = []
    monkeypatch.setattr(
        runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    _run(_cfg(), noise=EchoNoise())
    assert order == ["apply_final", "apply_final"]


def test_restore_is_retried_in_cleanup_when_it_fails(monkeypatch):
    order = []

    def exploding_final(self):
        order.append("apply_final")
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(runtime.LatentWalker, "apply_final", exploding_final)
    with pytest.raises(RuntimeError, match="restore exploded"):
        _run(_cfg(), noise=EchoNoise())
    assert order == ["apply_final", "apply_final"]


def test_forced_failure_restores_pristine_conditioning(monkeypatch, i2v_guider):
    """A mid-run failure restores the original conditioning latents."""
    calls = {"n": 0}
    expanded = runtime.spectral_expand

    def fake_expand(value, target_hw, sigma, seed):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced mid-run failure")
        return expanded(value, target_hw, sigma, seed)

    monkeypatch.setattr(runtime, "spectral_expand", fake_expand)
    with pytest.raises(RuntimeError, match="forced mid-run failure"):
        _run(_cfg(), i2v_guider, noise=EchoNoise())

    cond = i2v_guider.original_conds["positive"][0]
    for holder in cond["minimax_keyframes"] + cond["minimax_refs"]:
        assert holder["latent"].shape == (1, 1, 2, 8, 8), "conditioning not restored"


def test_i2v_smoke_and_pristine_restore_on_success(i2v_guider):
    original_conds = i2v_guider.original_conds
    cond = original_conds["positive"][0]
    keyframes, refs = cond["minimax_keyframes"], cond["minimax_refs"]
    out, denoised = _run(_ladder_cfg(3), i2v_guider)

    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    for holder in keyframes + refs:
        assert tuple(holder["latent"].shape[-2:]) == (8, 8)
    assert i2v_guider.original_conds is original_conds
    assert cond["minimax_keyframes"] is keyframes
    assert cond["minimax_refs"] is refs


# ---------------------------------------------------------------------------
# Image models (model_type="image")
# ---------------------------------------------------------------------------

IMAGE_CFG = _cfg(scales=(.5, 1.0), transition_steps=(5,))


def _run_image(guider, latent, cfg=IMAGE_CFG):
    return _run(cfg, guider, noise=SeededNoise(7), latent=latent, model_type="image")


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (1, 4, 1, 8, 8)])
def test_image_run_keeps_shape_and_carries_state_across_the_affine_latent_format(shape):
    clean = torch.randn((shape[0], 4, 8, 8), generator=torch.Generator().manual_seed(1))
    # A 5D single-frame latent comes from a 3-dimensional latent format (Wan 2.1, used by Krea 2).
    guider = OracleFlowGuider(clean, latent_dimensions=len(shape) - 2)
    out, denoised = _run_image(guider, image_latent(shape))
    assert out["samples"].shape == shape
    assert denoised["samples"].shape == shape
    assert [tuple(s.shape[-2:]) for s in guider.starts] == [(4, 4), (8, 8)]

    # The second stage starts from the first stage's end state, grown in DCT space and
    # rescaled by kappa: its low band is exactly kappa * that state.
    kappa, _ = aligned_sigma(float(SIGMAS[5]), 2.0)
    carried = lowpass_dct(guider.starts[1], (4, 4))
    assert torch.allclose(carried, kappa * guider.ends[0], atol=1e-5)

    # An ideal denoiser lands on the clean latent; the output is in latent-format space.
    expected = AffineModel().process_latent_out(clean).reshape(shape)
    assert torch.allclose(out["samples"], expected, atol=1e-5)


def test_coupled_full_grid_runs_for_image_latents():
    shape = (1, 4, 8, 8)
    out, _ = _run_image(
        OracleFlowGuider(torch.zeros(shape)), image_latent(shape),
        _cfg(scales=(.5, 1.0), transition_steps=(5,), noise_policy="coupled_full_grid"),
    )
    assert out["samples"].shape == shape


def test_runtime_fits_a_generic_empty_latent_to_the_model_first(monkeypatch):
    import comfy.sample

    seen = []

    def widen(model, latent_image, *ratios):
        seen.append(tuple(latent_image.shape))
        return torch.zeros(latent_image.shape[0], 4, *latent_image.shape[2:])

    monkeypatch.setattr(comfy.sample, "fix_empty_latent_channels", widen)
    out, _ = _run_image(OracleFlowGuider(torch.zeros(1, 4, 8, 8)), image_latent((1, 1, 8, 8)))
    assert seen == [(1, 1, 8, 8)]
    assert out["samples"].shape == (1, 4, 8, 8)
