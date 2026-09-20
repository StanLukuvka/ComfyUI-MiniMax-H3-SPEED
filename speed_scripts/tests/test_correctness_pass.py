"""Core runtime contracts for noise, outputs, I2V restore, and audio re-entry."""

import pytest
import torch

from conftest import make_fake_noise, make_latent, make_recording_guider
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import run_speed_pipeline


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def _run(config, *, guider=None, latent=None, sigmas=SIGMAS):
    return run_speed_pipeline(
        make_fake_noise(),
        guider if guider is not None else make_recording_guider(),
        sigmas,
        latent if latent is not None else make_latent(),
        config,
        sampler_override=object(),
        disable_pbar=True,
    )


def _cfg(**kwargs):
    return SpeedConfig(
        scales=(.33, .66, 1.0),
        transition_steps=(3, 5),
        transition_mode="explicit",
        **kwargs,
    )


def test_coupled_full_grid_projects_full_noise_in_the_spectral_domain():
    """Coupled stages project one full-resolution noise field into each stage grid.

    The expected re-entry noise is rebuilt from the same DCT operations used
    by the runtime and compared at both transitions.
    """
    import speed_scripts.spectral as spectral_mod
    from speed_scripts.flow import aligned_sigma, reentry_noise, to_internal_state

    torch.manual_seed(7)
    full_noise_video = torch.randn(1, 1, 2, 8, 8)  # the one full-grid field
    full_noise_audio = torch.zeros(1, 1, 2, 44)
    video_offset = 0.37  # makes the stage output (and source block) non-trivial

    class KnownNoise:
        seed = 42

        def generate_noise(self, latent):
            return type(
                "Nested",
                (),
                {"is_nested": True, "unbind": lambda self: [full_noise_video, full_noise_audio]},
            )()

    class AdditiveGuider:
        """Echo-plus-offset guider: public = pub + offset, records every pub."""

        class Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

            def process_latent_out(self, x):
                return x

        def __init__(self):
            self.model_patcher = type("P", (), {"model": self.Model()})()
            self.pubs = []

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            pub_video, pub_audio = noise.unbind()
            self.pubs.append(pub_video.clone())
            out_video = pub_video + video_offset
            out = type(
                "Nested",
                (),
                {"is_nested": True, "unbind": lambda self: [out_video, pub_audio]},
            )()
            if callback is not None:
                for i in range(len(sigmas) - 1):
                    callback(i, out, out, len(sigmas) - 1)
            return out

    def nested(video, audio):
        return type(
            "Nested", (), {"is_nested": True, "unbind": lambda self: [video, audio]},
        )()

    for temporal_scales in ((), (.5, .75, 1.0)):
        cfg = _cfg(noise_policy="coupled_full_grid", temporal_scales=temporal_scales)
        guider = AdditiveGuider()
        out, _ = run_speed_pipeline(
            KnownNoise(),
            guider,
            SIGMAS,
            make_latent(),
            cfg,
            sampler_override=object(),
            disable_pbar=True,
        )

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
        # Check stage sizes before comparing spectral re-entry.
        assert [p.shape[-3:] for p in guider.pubs] == thw

        # Full-resolution DCT shared by every coupled stage.
        full_dct = spectral_mod.dct2(spectral_mod.dct_temporal(full_noise_video))

        # --- Transition 1: stage 0 -> stage 1 -----------------------------
        s0_t, s0_h, s0_w = thw[0]
        # Stage 0 starts from the low-frequency block of the full noise field.
        pub0_video = spectral_mod.idct2(
            spectral_mod.dct2(full_noise_video[..., :s0_t, :, :])[..., :s0_h, :s0_w]
        )
        rsp_q0 = float(SIGMAS[3])
        internal0_video = to_internal_state(
            pub0_video + video_offset, full_noise_audio, rsp_q0, 4.0
        )[0]
        ratio0 = .66 / .33
        kappa0, new_q0 = aligned_sigma(rsp_q0, ratio0)
        t1, h1, w1 = thw[1]
        target_dct = full_dct[..., :t1, :h1, :w1] * rsp_q0
        target_dct[..., :s0_t, :s0_h, :s0_w] = spectral_mod.dct2(
            spectral_mod.dct_temporal(internal0_video)
        )
        expanded0 = spectral_mod.idct_temporal(spectral_mod.idct2(target_dct))
        expected_pub1 = reentry_noise(expanded0 * kappa0, new_q0)
        assert torch.allclose(guider.pubs[1], expected_pub1, atol=1e-4), (
            "transition 1 re-entry noise is not the spectral projection of "
            "the full-grid noise (pixel-space cropping drift?)"
        )

        # --- Transition 2: stage 1 -> stage 2 (full res) ------------------
        rsp_q1 = float(SIGMAS[5])
        internal1_video = to_internal_state(
            guider.pubs[1] + video_offset, full_noise_audio, rsp_q1, 4.0
        )[0]
        ratio1 = 1.0 / .66
        kappa1, new_q1 = aligned_sigma(rsp_q1, ratio1)
        t2, h2, w2 = thw[2]
        target_dct2 = full_dct[..., :t2, :h2, :w2] * rsp_q1
        target_dct2[..., :t1, :h1, :w1] = spectral_mod.dct2(
            spectral_mod.dct_temporal(internal1_video)
        )
        expanded1 = spectral_mod.idct_temporal(spectral_mod.idct2(target_dct2))
        expected_pub2 = reentry_noise(expanded1 * kappa1, new_q1)
        assert torch.allclose(guider.pubs[2], expected_pub2, atol=1e-4), (
            "transition 2 re-entry noise is not the spectral projection of "
            "the full-grid noise (pixel-space cropping drift?)"
        )

        final_video, _ = out["samples"].unbind()
        assert tuple(final_video.shape[-3:]) == (full_t, 8, 8)


def test_denoised_output_preserves_video_and_audio_streams():
    """Both returned LATENTs keep the full nested video+audio structure."""
    class RecordingModel:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0
        seen = []

        def process_latent_out(self, x):
            RecordingModel.seen.append(x)
            return x

    class Patcher:
        model = RecordingModel()

    guider = make_recording_guider()
    guider.model_patcher = Patcher()

    # x0_output carries the full nested x0 into process_latent_out.
    x0_output = {}
    out, denoised = run_speed_pipeline(
        make_fake_noise(),
        guider,
        SIGMAS,
        make_latent(),
        _cfg(),
        sampler_override=object(),
        disable_pbar=True,
        x0_output=x0_output,
    )

    assert RecordingModel.seen, "process_latent_out never ran (no x0 captured)"
    nested_x0 = RecordingModel.seen[-1]
    assert getattr(nested_x0, "is_nested", False), (
        "process_latent_out received a bare video tensor — the nested "
        "video+audio x0 was unpacked before reaching the model"
    )
    video, audio = nested_x0.unbind()
    assert video.ndim == 5 and audio.ndim == 4, "x0 streams unpacked wrong"

    # Both outputs keep the nested video and audio streams.
    for name, latent_out in (("output", out), ("denoised", denoised)):
        samples = latent_out["samples"]
        assert getattr(samples, "is_nested", False), f"{name} lost the nested bundle"
        v, a = samples.unbind()
        assert v.ndim == 5, f"{name} video stream missing/mangled"
        assert a.ndim == 4, f"{name} audio stream missing/mangled"


def test_forced_failure_restores_pristine_conditioning():
    """A mid-run failure restores the original conditioning latents."""
    cfg = _cfg()
    guider = make_recording_guider()

    # Use real I2V-shaped keyframe and reference holders.
    kf = {"latent": torch.zeros(1, 1, 2, 8, 8)}
    ref = {"latent": torch.zeros(1, 1, 2, 8, 8)}
    guider.original_conds = {
        "positive": [{"minimax_keyframes": [kf], "minimax_refs": [ref]}],
        "negative": [],
    }

    calls = {"n": 0}

    # Patch h3_runtime.spectral_expand, the binding used by run_speed_pipeline.
    import speed_scripts.h3_runtime as rt

    expanded = rt.spectral_expand

    def fake_expand(value, target_hw, sigma, seed):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("forced mid-run failure")
        return expanded(value, target_hw, sigma, seed)

    rt.spectral_expand = fake_expand
    try:
        with pytest.raises(RuntimeError, match="forced mid-run failure"):
            _run(cfg, guider=guider)
    finally:
        rt.spectral_expand = expanded

    # Conditioning returns to the original full resolution.
    assert kf["latent"].shape == (1, 1, 2, 8, 8), "keyframe not restored"
    assert ref["latent"].shape == (1, 1, 2, 8, 8), "ref not restored"


def test_audio_transition_oracle_clock_reindex_bridge():
    """Verify clock_reindex audio re-entry against the flow math."""
    from speed_scripts.flow import (
        aligned_sigma,
        clock_reindex_audio_state,
        reentry_noise,
        time_shift_sigma,
        to_internal_state,
    )

    sigmas = torch.tensor([1.0, .8, .6, .4, .2, 0.0])
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(2,),
        transition_mode="explicit",
        audio_policy="clock_reindex",
    )

    audio_out = 2.5  # non-zero constant the guider "denoises" to
    video_offset = 0.5

    class NoisyGuider:
        """Record each stage input and return a fixed non-zero audio stream."""

        class Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

            def process_latent_out(self, x):
                return x

        def __init__(self):
            self.model_patcher = type("P", (), {"model": self.Model()})()
            self.pubs = []

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            pub_video, pub_audio = noise.unbind()
            self.pubs.append((pub_video.clone(), pub_audio.clone()))
            out_video = pub_video + video_offset
            out_audio = torch.full_like(pub_audio, audio_out)
            out = type(
                "Nested",
                (),
                {"is_nested": True, "unbind": lambda self: [out_video, out_audio]},
            )()
            if callback is not None:
                for i in range(len(sigmas) - 1):
                    callback(i, out, out, len(sigmas) - 1)
            return out

    guider = NoisyGuider()
    x0_output = {}
    out, _ = run_speed_pipeline(
        make_fake_noise(),
        guider,
        sigmas,
        make_latent(),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
        x0_output=x0_output,
    )

    # Stage-1 re-entry audio carries non-zero state.
    assert len(guider.pubs) == 2
    reentry_video, reentry_audio = guider.pubs[1]
    assert torch.count_nonzero(reentry_audio) > 0, "oracle audio collapsed to zero"

    # --- Expected stage-1 re-entry audio --------------------------------
    video_shift, audio_shift = 12.0, 3.0
    audio_scale = video_shift / audio_shift  # 4.0 — as resolve_sigma_shifts computes it
    rsp_q = float(sigmas[2])  # stage-0 boundary (global index 2)
    ratio = 1.0 / .5
    kappa, new_q = aligned_sigma(rsp_q, ratio)

    # Convert the stage-0 public audio to the internal representation.
    stage0_pub_video, stage0_pub_audio = out_public = None, None
    # Recreate the fixed stage-0 audio output.
    stage0_pub_audio = torch.full_like(reentry_audio, audio_out)
    internal_audio = to_internal_state(
        torch.zeros_like(stage0_pub_audio), stage0_pub_audio, rsp_q, audio_scale
    )[1]
    assert torch.count_nonzero(internal_audio) > 0

    # x0 audio stays in model-output space.
    _, x0_audio = x0_output["x0"].unbind()
    assert torch.count_nonzero(x0_audio) > 0, "x0 audio must be non-zero"

    old_audio_sigma = time_shift_sigma(rsp_q, video_shift, audio_shift)
    new_audio_sigma = time_shift_sigma(new_q, video_shift, audio_shift)
    transitioned_audio = clock_reindex_audio_state(
        internal_audio,
        x0_audio,
        rsp_q,
        new_q,
        old_audio_sigma,
        new_audio_sigma,
        audio_scale,
    )
    # Convert the transitioned audio back to the public sigma-scaled form.
    expected_reentry_audio = reentry_noise(transitioned_audio, new_q)

    assert torch.allclose(reentry_audio, expected_reentry_audio, atol=1e-5), (
        "stage-1 re-entry audio does not match the independent clock_reindex "
        "oracle (audio_scale=4.0)"
    )

    # Final output keeps the non-zero audio stream.
    _, final_audio = out["samples"].unbind()
    assert torch.count_nonzero(final_audio) > 0
    assert final_audio.shape == reentry_audio.shape

