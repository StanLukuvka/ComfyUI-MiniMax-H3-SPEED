"""run_speed_pipeline end to end: H3 and image runs, noise policies, audio, I2V, failures."""

import pytest
import torch

from conftest import (
    SIGMAS,
    AffineModel,
    EchoGuider,
    EchoNoise,
    NestedTensor,
    OracleFlowGuider,
    SeededNoise,
    StubSampler,
    aligned,
    expected_stage_sigmas,
    image_latent,
    make_latent,
)
from speed_scripts import runtime, spectral
from speed_scripts.config import SpeedConfig
from speed_scripts.runtime import run_speed_pipeline

QUARTERS = (.25, .5, .75, 1.0)


def _run(config, guider=None, *, noise=None, latent=None, sigmas=SIGMAS, **kwargs):
    kwargs.setdefault("sampler", StubSampler())
    kwargs.setdefault("model_type", "h3_video")
    return run_speed_pipeline(
        noise if noise is not None else SeededNoise(),
        guider if guider is not None else EchoGuider(),
        sigmas,
        latent if latent is not None else make_latent(),
        config,
        disable_pbar=True,
        **kwargs,
    )


def _cfg(scales=(.33, .66, 1.0), transition_steps=(3, 5), **kwargs):
    return SpeedConfig(scales=scales, transition_steps=transition_steps, transition_mode="explicit", **kwargs)


def _assert_full_res_nested(latent):
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 1, 2, 8, 8) and audio.shape == (1, 1, 2, 44)


@pytest.mark.parametrize("noise_policy", ["direct_coarse", "coupled_full_grid"])
def test_run_follows_the_stage_ladder_on_one_global_timeline(noise_policy):
    sampler = StubSampler("heun")
    guider = EchoGuider(video_offset=.5)
    previews = []
    out, denoised = _run(
        _cfg(QUARTERS, (3, 5, 8), noise_policy=noise_policy), guider, sampler=sampler,
        preview_callback=lambda step, x0, x, total: previews.append((step, total)),
    )

    # Global boundaries; each re-entry sigma is aligned by the scale ratio.
    for actual, wanted in zip(guider.sigma_calls, expected_stage_sigmas(SIGMAS, (3, 5, 8), QUARTERS), strict=True):
        assert actual == pytest.approx(wanted)
    assert guider.latent_sizes == [(2, 2), (4, 4), (6, 6), (8, 8)]
    assert [tuple(video.shape) for video, _ in guider.noise] == [(1, 1, 2, s, s) for s in (2, 4, 6, 8)]
    assert all(seen is sampler for seen in guider.samplers)
    assert guider.seeds == [42] * 4
    # Progress and previews see one run, not one per stage.
    assert previews == [(i, 10) for i in range(10)]

    # Both outputs and the x0 handed to process_latent_out keep the nested video+audio latent.
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    assert out["samples"].unbind()[0].abs().sum() > 0
    x0 = guider.model_patcher.model.latents_out[-1]
    assert getattr(x0, "is_nested", False) and [p.ndim for p in x0.unbind()] == [5, 4]


_SPREAD_FIT = dict(delta=.01, noise_amplitude=219.48, noise_decay_exponent=2.42)


@pytest.mark.parametrize(
    "scales,fit,steps,hw,boundaries",
    [
        ((.25, .5, 1.0), _SPREAD_FIT, 20, (44, 80), (1, 2)),
        # Coincident boundaries align the same sigma twice; the middle stage has no steps.
        ((.25, .5, 1.0), _SPREAD_FIT, 10, (8, 8), (1, 1)),
        # H3's Euler calibration on a tiny latent: two zero-step stages in a row.
        (QUARTERS, dict(delta=.005, noise_amplitude=12.105, noise_decay_exponent=.773), 10, (8, 8), (1, 1, 1)),
    ],
    ids=["spread", "coincident", "h3_euler_four_stage"],
)
def test_delta_custom_places_boundaries_from_the_power_spectrum(scales, fit, steps, hw, boundaries):
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    cfg = SpeedConfig(scales=scales, transition_steps=(), transition_mode="delta_custom", **fit)
    guider = EchoGuider()
    out, _ = _run(cfg, guider, latent=make_latent(h=hw[0], w=hw[1]), sigmas=sigmas)

    for actual, wanted in zip(guider.sigma_calls, expected_stage_sigmas(sigmas, boundaries, scales), strict=True):
        assert actual == pytest.approx(wanted)
    assert tuple(out["samples"].unbind()[0].shape[-2:]) == hw


SIGMA_ONE_BOUNDARY = torch.tensor([1.0, 1.0, .5, 0.0])


@pytest.mark.parametrize(
    "latent,config,sigmas,image,error",
    [
        ({**make_latent(), "noise_mask": torch.ones(1)}, _cfg(), SIGMAS, False, "noise masks"),
        ({"samples": NestedTensor([torch.ones(1, 1, 2, 8, 8), torch.zeros(1, 1, 2, 44)])},
         _cfg(), SIGMAS, False, "empty input latent"),
        (make_latent(), _cfg(), SIGMAS[:2], False, "one-dimensional schedule"),
        (make_latent(), _cfg((.5, 1.0), (10,)), SIGMAS, False, "inside the sigma schedule"),
        # A boundary at sigma 1 cannot be carried into the next stage.
        (make_latent(), _cfg((.5, 1.0), (1,)), SIGMA_ONE_BOUNDARY, False, r"endpoint sigma must be in \[0, 1\)"),
        (image_latent((1, 4, 8, 8)), _cfg((.5, 1.0), (1,)), SIGMA_ONE_BOUNDARY, True,
         "transition sigma must be between zero and one"),
    ],
    ids=["noise_mask", "non_empty", "short_sigmas", "boundary_at_end", "video_boundary_at_sigma_one",
         "image_boundary_at_sigma_one"],
)
def test_run_rejects_unsupported_inputs(latent, config, sigmas, image, error):
    guider = OracleFlowGuider(torch.zeros(1, 4, 8, 8)) if image else None
    with pytest.raises(ValueError, match=error):
        _run(config, guider, latent=latent, sigmas=sigmas, model_type="flow_image" if image else "h3_video")


@pytest.mark.parametrize("temporal_scales", [(), (.5, .75, 1.0)], ids=["spatial", "temporal"])
def test_coupled_full_grid_reuses_one_full_noise_field_in_the_spectral_domain(monkeypatch, temporal_scales):
    """Each stage's re-entry noise is the carried state with the full field's coefficients around it."""
    dct, idct = (lambda x: spectral.dct2(spectral.dct_temporal(x))), (lambda c: spectral.idct_temporal(spectral.idct2(c)))
    full_grid_transforms = []
    original = runtime.dct_temporal

    def counting_dct_temporal(value):
        if tuple(value.shape[-2:]) == (8, 8):
            full_grid_transforms.append(value.shape)
        return original(value)

    monkeypatch.setattr(runtime, "dct_temporal", counting_dct_temporal)
    noise = SeededNoise(7)
    full_video, _ = noise.generate_noise(make_latent()).unbind()
    offset = 0.37  # makes each stage output (the carried block) non-trivial
    guider = EchoGuider(video_offset=offset)
    _run(_cfg(noise_policy="coupled_full_grid", temporal_scales=temporal_scales), guider, noise=noise)
    pubs = [video for video, _ in guider.noise]

    thw = [(max(1, round(2 * temporal_scales[i])) if temporal_scales else 2, round(8 * s), round(8 * s))
           for i, s in enumerate((.33, .66, 1.0))]
    assert [tuple(p.shape[-3:]) for p in pubs] == thw
    # Stage 0 starts from the low band of the full field.
    t0, h0, w0 = thw[0]
    assert torch.allclose(pubs[0], spectral.idct2(spectral.dct2(full_video[..., :t0, :, :])[..., :h0, :w0]), atol=1e-5)
    full = dct(full_video)
    for i, (boundary, ratio) in enumerate(((3, .66 / .33), (5, 1 / .66))):
        q = float(SIGMAS[boundary])
        kappa, new_q = aligned(q, ratio)
        (st, sh, sw), (t, h, w) = thw[i], thw[i + 1]
        coefficients = full[..., :t, :h, :w] * q
        coefficients[..., :st, :sh, :sw] = dct((pubs[i] + offset) * (1 - q))
        assert torch.allclose(pubs[i + 1], idct(coefficients) * kappa / new_q, atol=1e-4)
    assert len(full_grid_transforms) == 1  # transformed once per generation


def test_h3_audio_crosses_a_transition_on_its_own_sigma_clock():
    """clock_reindex: re-noise the carried audio from its x0 on the audio clock (shifts 12 and 3)."""
    sigmas = torch.tensor([1.0, .8, .6, .4, .2, 0.0])
    audio_out = 2.5
    guider = EchoGuider(video_offset=.5, audio_value=audio_out)
    # The H3 shifts on the model win over a generic sampling shift.
    guider.model_patcher.objects["model_sampling"] = type("Sampling", (), {"shift": 1.0, "audio_shift": 1.0})()
    x0_output = {}
    out, _ = _run(_cfg((.5, 1.0), (2,)), guider, noise=EchoNoise(), sigmas=sigmas, x0_output=x0_output)

    def audio_sigma(sigma, video_shift=12.0, audio_shift=3.0):
        base = sigma / (video_shift + sigma * (1.0 - video_shift))
        return audio_shift * base / (1.0 + (audio_shift - 1.0) * base)

    q = float(sigmas[2])
    _, new_q = aligned(q, 2.0)
    old_a, new_a = audio_sigma(q), audio_sigma(new_q)
    audio_scale = 12.0 / 3.0
    _, x0_audio = x0_output["x0"].unbind()
    carried = torch.full_like(x0_audio, audio_out) * audio_scale * (1 - q)  # internal state at q
    clean = x0_audio / audio_scale
    native_noise = (carried * old_a / q - (1 - old_a) * clean) / old_a
    expected = ((1 - new_a) * clean + new_a * native_noise) * new_q / new_a / new_q

    _, reentry_audio = guider.noise[1]
    assert torch.count_nonzero(reentry_audio) > 0
    assert torch.allclose(reentry_audio, expected, atol=1e-5)
    assert torch.count_nonzero(out["samples"].unbind()[1]) > 0


def attach_i2v(guider):
    """Give ``guider`` random 8x8 I2V keyframes (positive and negative) and refs; return both lists."""
    keyframes = [{"latent": torch.rand(1, 1, 2, 8, 8)} for _ in range(3)]
    refs = [{"latent": torch.rand(1, 1, 2, 8, 8)} for _ in range(2)]
    guider.original_conds = {
        "positive": [{"minimax_keyframes": keyframes[:2], "minimax_refs": refs}],
        "negative": [{"minimax_keyframes": keyframes[2:], "minimax_refs": []}],
    }
    return keyframes, refs


def _stage_resize(latent, size):
    """Bilinear per-frame resize of a [B,C,T,H,W] latent, as H3 conditioning is resized."""
    b, c, t, h, w = latent.shape
    flat = latent.transpose(1, 2).reshape(b * t, c, h, w)
    out = torch.nn.functional.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return out.reshape(b, t, c, *size).transpose(1, 2)


def test_i2v_keyframes_follow_each_stage_from_pristine_and_are_restored_for_the_final_stage():
    guider = EchoGuider(video_offset=.5)
    keyframes, refs = attach_i2v(guider)
    pristine = [holder["latent"].clone() for holder in keyframes + refs]
    seen = []
    sample = guider.sample

    def recording_sample(*args, **kwargs):
        seen.append([holder["latent"].clone() for holder in keyframes + refs])
        return sample(*args, **kwargs)

    guider.sample = recording_sample
    attributes, conds = set(vars(guider)), guider.original_conds
    _run(_cfg((1 / 3, 2 / 3, 1.0)), guider)

    # Stages are 3x3, 5x5, 8x8; keyframes use H3's even patch grid (4x4, 6x6) and are always
    # resized from the pristine latent; refs never change.
    for stage, size in enumerate([(4, 4), (6, 6), None]):
        for index, latent in enumerate(seen[stage]):
            is_keyframe = index < len(keyframes)
            wanted = _stage_resize(pristine[index], size) if is_keyframe and size else pristine[index]
            assert torch.allclose(latent, wanted, atol=1e-6)
    for holder, original in zip(keyframes + refs, pristine):
        assert torch.equal(holder["latent"], original)
    # The walker is local to the run: nothing is left on the guider.
    assert set(vars(guider)) == attributes and guider.original_conds is conds


def test_failures_stop_cleanly_and_the_next_generation_starts_clean():
    # A failing preview callback is disabled for the rest of the run; the run completes.
    previews = []

    def failing_preview(step, x0, x, total):
        previews.append(step)
        raise RuntimeError("preview exploded")

    _run(_cfg(QUARTERS, (3, 5, 8)), preview_callback=failing_preview)
    assert previews == [0]

    # A failing stage stops the run and restores the I2V conditioning.
    guider = EchoGuider(video_offset=.5, fail_at_call=2)
    keyframes, refs = attach_i2v(guider)
    pristine = [holder["latent"].clone() for holder in keyframes + refs]
    with pytest.raises(RuntimeError, match="stage exploded"):
        _run(_cfg(), guider)
    assert len(guider.samplers) == 2  # the final stage never ran
    for holder, original in zip(keyframes + refs, pristine):
        assert torch.equal(holder["latent"], original)

    # The same guider's next generation matches a fresh run exactly.
    again, _ = _run(_cfg(), guider)
    control, _ = _run(_cfg(), EchoGuider(video_offset=.5))
    for a, b in zip(again["samples"].unbind(), control["samples"].unbind()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("noise_policy", ["direct_coarse", "coupled_full_grid"])
@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (1, 4, 1, 8, 8)])
def test_image_run_carries_state_across_the_affine_latent_format(shape, noise_policy):
    clean = torch.randn((shape[0], 4, 8, 8), generator=torch.Generator().manual_seed(1))
    # A 5D single-frame latent comes from a 3-dimensional latent format (Wan 2.1, used by Krea 2).
    guider = OracleFlowGuider(clean, latent_dimensions=len(shape) - 2)
    out, denoised = _run(
        _cfg((.5, 1.0), (5,), noise_policy=noise_policy), guider, noise=SeededNoise(7),
        latent=image_latent(shape), model_type="flow_image",
    )
    assert out["samples"].shape == denoised["samples"].shape == shape
    assert [tuple(s.shape[-2:]) for s in guider.starts] == [(4, 4), (8, 8)]

    # Stage 2 starts from stage 1's end state grown in DCT space and rescaled by kappa:
    # its low band is exactly kappa * that state.
    kappa, _ = aligned(float(SIGMAS[5]), 2.0)
    assert torch.allclose(spectral.lowpass_dct(guider.starts[1], (4, 4)), kappa * guider.ends[0], atol=1e-5)
    # An ideal denoiser lands on the clean latent; the output is in latent-format space.
    assert torch.allclose(out["samples"], AffineModel().process_latent_out(clean).reshape(shape), atol=1e-5)


def test_image_run_fits_a_generic_empty_latent_to_the_model_first(monkeypatch):
    import comfy.sample

    seen = []

    def widen(model, latent_image, *ratios):
        seen.append(tuple(latent_image.shape))
        return torch.zeros(latent_image.shape[0], 4, *latent_image.shape[2:])

    monkeypatch.setattr(comfy.sample, "fix_empty_latent_channels", widen)
    out, _ = _run(_cfg((.5, 1.0), (5,)), OracleFlowGuider(torch.zeros(1, 4, 8, 8)),
                  latent=image_latent((1, 1, 8, 8)), model_type="flow_image")
    assert seen == [(1, 1, 8, 8)]
    assert out["samples"].shape == (1, 4, 8, 8)
