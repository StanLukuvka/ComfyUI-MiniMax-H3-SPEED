"""Behavioral contracts for the focused correctness pass.

Covers the four runtime invariants that must hold before the scheduler
rework, without touching scheduling logic itself:

1. Coupled noise follows the configured stage ladder — each transition
   expands only to the NEXT stage's grid (spatial and 3D coupled paths).
2. `denoised_output` preserves H3's full nested video+audio latent.
3. The I2V LatentWalker lifecycle is exception-safe: any mid-run failure
   restores pristine conditioning latents and drops the walker.
4. The `clock_reindex_audio_state` sigma bridge is exercised end-to-end
   through a real transition (audio-transition oracle).
"""

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
    """Coupled transitions = spectral projection of the SAME full-grid noise.

    For every transition, the runtime must derive the next stage's coupled
    noise by taking the combined temporal+spatial DCT of the ORIGINAL
    full-resolution noise and keeping the low-frequency coefficient block
    for (next_t, next_h, next_w) — NOT by cropping the noise in pixel/latent
    space (which changes the DCT spectrum). The test recomputes both
    transitions' re-entry noise independently from the spectral primitives
    and asserts numerical equality with what the runtime hands the next
    stage's guider call. Parametrized over the spatial-only path (constant
    temporal block) and the 3D path (temporal block grows 1→2).
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
        # Stage grids from the same rounding the runtime uses.
        thw = [
            (
                full_t if not temporal_scales else max(1, round(full_t * temporal_scales[i])),
                max(1, round(8 * s)),
                max(1, round(8 * s)),
            )
            for i, s in enumerate((.33, .66, 1.0))
        ]
        # Ladder sanity (geometry), then the real spectral oracle below.
        assert [p.shape[-3:] for p in guider.pubs] == thw

        # The single full-grid field the whole run coupled from.
        full_dct = spectral_mod.dct2(spectral_mod.dct_temporal(full_noise_video))

        # --- Transition 1: stage 0 -> stage 1 -----------------------------
        s0_t, s0_h, s0_w = thw[0]
        # Stage-0 pub: coupled coarse noise = spatial lowpass of the full
        # field's low block (temporal pixel-crop only when t0 < full_t).
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

        # Final output video is the guider's final stage return, full res.
        final_video, _ = out["samples"].unbind()
        assert tuple(final_video.shape[-3:]) == (full_t, 8, 8)


def test_denoised_output_preserves_video_and_audio_streams():
    """Both returned LATENTs must keep H3's nested video+audio structure.

    The old code extracted only the video stream from the step-x0 and fed
    that through process_latent_out, so `denoised` lost the audio stream.
    """
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

    # capture_state is shared with the runtime's fallback wrapper: the fake
    # guider's callback writes x0 into it, and `denoised` is reconstructed
    # from x0_output["x0"] — assert the FULL nested x0 arrives at
    # process_latent_out, not a video-only extraction.
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

    # Both outputs are well-formed LATENT dicts whose samples are the full
    # nested bundle (video AND audio streams present).
    for name, latent_out in (("output", out), ("denoised", denoised)):
        samples = latent_out["samples"]
        assert getattr(samples, "is_nested", False), f"{name} lost the nested bundle"
        v, a = samples.unbind()
        assert v.ndim == 5, f"{name} video stream missing/mangled"
        assert a.ndim == 4, f"{name} audio stream missing/mangled"


def test_forced_failure_restores_pristine_and_drops_walker():
    """A mid-run failure must restore pristine conds and remove the walker.

    Forces the failure inside the transition (spectral expansion of stage
    1), after stage 0 has already downscaled the keyframe latents — the
    exact window where the pre-try/finally code leaked half-resized conds
    and a stashed `_speed_latent_walker`.
    """
    cfg = _cfg()
    guider = make_recording_guider()

    # Give the guider real I2V-shaped conditioning so the walker has
    # keyframe/ref latents to snapshot and resize.
    kf = {"latent": torch.zeros(1, 1, 2, 8, 8)}
    ref = {"latent": torch.zeros(1, 1, 2, 8, 8)}
    guider.original_conds = {
        "positive": [{"minimax_keyframes": [kf], "minimax_refs": [ref]}],
        "negative": [],
    }

    calls = {"n": 0}

    # The runtime imports spectral_expand by name (from .spectral import ...),
    # so patch the name where the runtime actually looks it up.
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

    from speed_scripts.h3_runtime import _LW_ATTR

    # 1. The walker was removed from the guider.
    assert not hasattr(guider, _LW_ATTR), "walker left stashed on the guider"
    # 2. Conditioning latents are back at pristine full res (not the
    #    half-res tensors stage 0 had installed).
    assert kf["latent"].shape == (1, 1, 2, 8, 8), "keyframe not restored"
    assert ref["latent"].shape == (1, 1, 2, 8, 8), "ref not restored"


def test_audio_transition_oracle_clock_reindex_bridge():
    """Numeric oracle for the clock_reindex sigma bridge.

    A custom guider produces known NON-ZERO public audio and non-zero x0
    audio. The test independently computes the expected stage-1 re-entry
    audio from the flow primitives — to_internal_state, time_shift_sigma,
    clock_reindex_audio_state (audio_scale = video_shift / audio_shift =
    12/3 = 4.0), reentry_noise — and asserts numerical equality with the
    audio stream the runtime actually hands the stage-1 guider call.
    """
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
        """Records the incoming pub (noise slot) per stage; returns non-zero audio.

        The runtime calls guider.sample(pub_noise, zero_latent, ...): the
        carried re-entry state arrives in the NOISE slot, the zero latent in
        latent_image. So the audio the runtime hands stage 1 is the noise
        slot's audio stream — that is what the oracle must match.
        """

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

    # Geometry + non-triviality: the stage-1 re-entry audio must be non-zero.
    assert len(guider.pubs) == 2
    reentry_video, reentry_audio = guider.pubs[1]
    assert torch.count_nonzero(reentry_audio) > 0, "oracle audio collapsed to zero"

    # --- Independent oracle for the stage-1 re-entry audio --------------
    video_shift, audio_shift = 12.0, 3.0
    audio_scale = video_shift / audio_shift  # 4.0 — as resolve_sigma_shifts computes it
    rsp_q = float(sigmas[2])  # stage-0 boundary (global index 2)
    ratio = 1.0 / .5
    kappa, new_q = aligned_sigma(rsp_q, ratio)

    # Carried audio entering the transition: the runtime converts the stage-0
    # PUBLIC output (the guider's return) to the internal representation.
    stage0_pub_video, stage0_pub_audio = out_public = None, None
    # Recreate stage-0's public output exactly as the runtime saw it: the
    # guider returned (noise video + offset, audio_out flat).
    stage0_pub_audio = torch.full_like(reentry_audio, audio_out)
    internal_audio = to_internal_state(
        torch.zeros_like(stage0_pub_audio), stage0_pub_audio, rsp_q, audio_scale
    )[1]
    assert torch.count_nonzero(internal_audio) > 0

    # Clean audio = x0 audio, passed RAW (the runtime does not convert it).
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
    # The runtime re-enters stage 1 with reentry_noise(audio, new_q): the
    # public (sigma-scaled) representation the guider receives in the noise slot.
    expected_reentry_audio = reentry_noise(transitioned_audio, new_q)

    assert torch.allclose(reentry_audio, expected_reentry_audio, atol=1e-5), (
        "stage-1 re-entry audio does not match the independent clock_reindex "
        "oracle (audio_scale=4.0)"
    )

    # And the final output audio is the guider's last stage return — audio
    # stream survived to the end with non-zero content.
    _, final_audio = out["samples"].unbind()
    assert torch.count_nonzero(final_audio) > 0
    assert final_audio.shape == reentry_audio.shape


def test_successful_run_leaves_no_walker_on_guider():
    """Happy path: the run-scoped finally must also drop the walker."""
    from speed_scripts.h3_runtime import _LW_ATTR

    guider = make_recording_guider()
    _run(_cfg(), guider=guider)
    assert not hasattr(guider, _LW_ATTR)
