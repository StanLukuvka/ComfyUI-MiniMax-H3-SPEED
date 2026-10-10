"""``upscale_type`` contracts (dct | bilinear | bicubic | nearest-exact).

Test list per the approved plan (1-7):
1.  Config validation and default.
2.  Builder / node passthrough (widget last, default dct, value reaches SpeedConfig).
3.  DCT regression: ``upscale_type="dct"`` output is bit-identical to the
    pre-change reference tensor (speed_scripts/tests/data).
4.  Space round-trip: zero noise, ``x0_hat == x`` at ``q -> 0`` gives
    ``new_x == x0_up``.
5.  Re-noise math on ``_renoise_upscale`` (shape/dtype, no 1/ratio loss,
    noise level, determinism, coupled path, frame non-mixing, T guard).
6.  Pipeline smoke per mode (full-res shape, audio unaffected).
7.  Missing x0 raises closed.
"""

import pytest
import torch

from conftest import (
    RecordingEchoGuider,
    SeededRandomNoise,
    make_fake_noise,
    make_latent,
    make_nested,
    make_recording_guider,
)
from speed_scripts.config import UPSCALE_TYPES, SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import _renoise_upscale, run_speed_pipeline
from speed_scripts.planning import build_automatic_speed_config


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

def test_upscale_type_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported upscale_type"):
        SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), upscale_type="bogus")


@pytest.mark.parametrize("mode", UPSCALE_TYPES)
def test_upscale_type_config_accepts_every_supported_mode(mode):
    config = SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), upscale_type=mode)
    assert config.upscale_type == mode


def test_upscale_type_config_defaults_to_dct():
    assert SpeedConfig().upscale_type == "dct"


# ---------------------------------------------------------------------------
# 2. Builder / node passthrough
# ---------------------------------------------------------------------------

def test_automatic_input_types_expose_upscale_type_last_with_dct_default():
    import importlib

    node = importlib.import_module("sampler_node").MiniMaxH3SPEEDSampler
    required = node.INPUT_TYPES()["required"]
    keys = list(required)
    assert keys[-1] == "upscale_type"
    assert required["upscale_type"][0] == list(UPSCALE_TYPES)
    assert required["upscale_type"][1]["default"] == "dct"


def test_builder_forwards_upscale_type_to_speed_config():
    config = build_automatic_speed_config(
        stages=3,
        noise_policy="direct_coarse",
        delta=0.005,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
        upscale_type="bilinear",
    )
    assert config.upscale_type == "bilinear"
    assert config.transition_mode == "delta_custom"


def test_builder_defaults_upscale_type_to_dct():
    config = build_automatic_speed_config(
        stages=2,
        noise_policy="direct_coarse",
        delta=0.005,
        noise_amplitude=12.105,
        noise_decay_exponent=0.773,
        seed_offset=10000,
    )
    assert config.upscale_type == "dct"


def test_node_value_reaches_speed_config(monkeypatch):
    import importlib

    mod = importlib.import_module("sampler_node")
    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kwargs):
        captured["config"] = config
        return latent_image, latent_image

    monkeypatch.setattr(mod, "run_speed_pipeline", fake_pipeline)
    mod.MiniMaxH3SPEEDSampler().sample(
        noise=object(),
        guider=type("Guider", (), {})(),
        sigmas=None,
        latent_image=make_latent(h=8, w=8),
        sampler_name="euler",
        stages=3,
        upscale_type="nearest-exact",
    )
    assert captured["config"].upscale_type == "nearest-exact"


def test_node_rejects_upscale_type_not_in_widget_list():
    import importlib

    mod = importlib.import_module("sampler_node")
    with pytest.raises(ValueError, match="unsupported upscale_type"):
        mod.MiniMaxH3SPEEDSampler().sample(
            noise=object(),
            guider=type("Guider", (), {})(),
            sigmas=None,
            latent_image=make_latent(h=8, w=8),
            sampler_name="euler",
            stages=3,
            upscale_type="catmull_rom",
        )


# ---------------------------------------------------------------------------
# 3. DCT regression (bit-identical to the pre-change output)
# ---------------------------------------------------------------------------

def test_default_dct_pipeline_is_bit_identical_to_pre_change_reference():
    reference = torch.load(
        __file__.rsplit("/", 1)[0] + "/data/upscale_dct_reference.pt",
        weights_only=True,
    )
    sigmas = torch.linspace(1.0, 0.0, 13)
    cfg = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(4,),
        transition_mode="explicit",
        noise_policy="direct_coarse",
    )
    out, _ = run_speed_pipeline(
        SeededRandomNoise(seed=42),
        RecordingEchoGuider(video_offset=0.37),
        sigmas,
        make_latent(h=8, w=8, t=2),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
        x0_output={},
    )
    video, audio = out["samples"].unbind()
    assert torch.equal(video, reference["video"])
    assert torch.equal(audio, reference["audio"])


def test_explicit_dct_choice_matches_default_pipeline_bit_for_bit():
    """The string 'dct' selects the unchanged legacy code path."""
    sigmas = torch.linspace(1.0, 0.0, 13)
    cfg = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(4,),
        transition_mode="explicit",
        noise_policy="direct_coarse",
        upscale_type="dct",
    )
    out, _ = run_speed_pipeline(
        SeededRandomNoise(seed=42),
        RecordingEchoGuider(video_offset=0.37),
        sigmas,
        make_latent(h=8, w=8, t=2),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
        x0_output={},
    )
    reference = torch.load(
        __file__.rsplit("/", 1)[0] + "/data/upscale_dct_reference.pt",
        weights_only=True,
    )
    video, audio = out["samples"].unbind()
    assert torch.equal(video, reference["video"])
    assert torch.equal(audio, reference["audio"])


# ---------------------------------------------------------------------------
# 4. Space round-trip (zero noise)
# ---------------------------------------------------------------------------

def test_zero_noise_round_trip_matches_plain_upscale():
    """BUG-WATCH: x0's video stream must be consumed as clean video, un-scaled.

    At q -> 0 (sigma = 0, x0_hat == x): new_x = (1 - 0) * x0_up + 0 * eps,
    and after reentry_noise(s, sigma) division at the same sigma the public
    re-entry equals x0_up divided by sigma — but the internal transitioned
    state itself must equal x0_up exactly. This test pins the internal call.
    """
    x0_video = torch.arange(1 * 1 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 1, 2, 4, 4)
    x0_video = x0_video / x0_video.abs().max()  # normalized, non-trivial

    out = _renoise_upscale(
        x0_video,
        target_thw=(2, 8, 8),
        sigma=0.0,
        mode="bilinear",
        seed=0,
    )
    # sigma = 0: new_x == x0_up exactly (interpolate, no noise contribution).
    from speed_scripts.h3_runtime import _interpolate_video

    expected = _interpolate_video(x0_video, (8, 8), "bilinear")
    assert torch.equal(out, expected)


# ---------------------------------------------------------------------------
# 5. Re-noise math (unit, on _renoise_upscale)
# ---------------------------------------------------------------------------

def test_renoise_upscale_output_shape_and_dtype():
    x0_video = torch.randn(1, 2, 3, 6, 8, dtype=torch.float32)
    out = _renoise_upscale(
        x0_video,
        target_thw=(3, 12, 16),
        sigma=0.5,
        mode="bicubic",
        seed=7,
    )
    assert tuple(out.shape) == (1, 2, 3, 12, 16)
    assert out.dtype == torch.float32


def test_renoise_upscale_constant_signal_amplitude_is_preserved():
    """No 1/ratio amplitude loss: interpolation preserves the DC level.

    At sigma = 0 the output is exactly the upscaled clean signal, so a
    constant x0 must stay that same constant (the DCT zero-pad, by contrast,
    scales the signal by 1/ratio before kappa compensation).
    """
    for mode in ("bilinear", "bicubic", "nearest-exact"):
        constant = torch.full((1, 1, 2, 4, 4), 0.7)
        out = _renoise_upscale(
            constant,
            target_thw=(2, 8, 8),
            sigma=0.0,
            mode=mode,
            seed=1,
        )
        assert out.shape == (1, 1, 2, 8, 8)
        assert torch.allclose(out, torch.full_like(out, 0.7), atol=1e-4)


@pytest.mark.parametrize("mode", ("bilinear", "bicubic", "nearest-exact"))
def test_renoise_upscale_zero_signal_gives_white_noise_at_sigma(mode):
    """With x0 = 0, output std must approximate sigma (white, fresh noise)."""
    torch.manual_seed(0)
    x0_video = torch.zeros(1, 1, 2, 4, 4)
    sigma = 0.8
    out = _renoise_upscale(
        x0_video,
        target_thw=(2, 16, 16),
        sigma=sigma,
        mode=mode,
        seed=3,
    )
    assert out.shape == (1, 1, 2, 16, 16)
    std = out.std().item()
    assert abs(std - sigma) < 0.05, f"noise std {std} far from sigma {sigma}"


def test_renoise_upscale_same_seed_identical_and_stage_seed_differs():
    x0_video = torch.randn(1, 1, 2, 4, 4)
    kwargs = dict(target_thw=(2, 8, 8), sigma=0.6, mode="bilinear")
    same_a = _renoise_upscale(x0_video, seed=100, **kwargs)
    same_b = _renoise_upscale(x0_video, seed=100, **kwargs)
    other = _renoise_upscale(x0_video, seed=101, **kwargs)
    assert torch.equal(same_a, same_b)
    assert not torch.equal(same_a, other)


def test_renoise_upscale_coupled_path_matches_hand_built_eps():
    """coupled_full_grid: epsilon is the truncated shared coefficient block
    inverse-transformed (idct_temporal(idct2(...)))."""
    from speed_scripts.spectral import dct2, dct_temporal, idct2, idct_temporal

    torch.manual_seed(11)
    full_noise_video = torch.randn(1, 1, 2, 8, 8)
    full_noise_coefficients = dct2(dct_temporal(full_noise_video))

    x0_video = torch.randn(1, 1, 2, 4, 4)
    sigma, seed = 0.5, 9

    # Hand-built expectation: truncate the shared field, invert, mix.
    coeff_block = full_noise_coefficients[..., :2, :8, :8]
    eps = idct_temporal(idct2(coeff_block))
    x0_up = torch.nn.functional.interpolate(
        x0_video.transpose(1, 2).reshape(2, 1, 4, 4),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 2, 1, 8, 8).transpose(1, 2)
    expected = ((1.0 - sigma) * x0_up + sigma * eps).to(x0_video.dtype)

    out = _renoise_upscale(
        x0_video,
        target_thw=(2, 8, 8),
        sigma=sigma,
        mode="bilinear",
        seed=seed,
        full_noise_coefficients=full_noise_coefficients,
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_renoise_upscale_frames_are_not_mixed():
    """A latent whose frames differ stays per-frame distinct after upscale."""
    frame_a = torch.full((1, 1, 1, 4, 4), -1.0)
    frame_b = torch.full((1, 1, 1, 4, 4), 1.0)
    x0_video = torch.cat([frame_a, frame_b], dim=2)
    out = _renoise_upscale(
        x0_video,
        target_thw=(2, 8, 8),
        sigma=0.0,
        mode="bilinear",
        seed=0,
    )
    assert torch.all(out[:, :, 0] < 0)
    assert torch.all(out[:, :, 1] > 0)


def test_renoise_upscale_rejects_frame_count_change():
    x0_video = torch.randn(1, 1, 2, 4, 4)
    with pytest.raises(ValueError, match="cannot change the frame count"):
        _renoise_upscale(
            x0_video,
            target_thw=(4, 8, 8),
            sigma=0.5,
            mode="bilinear",
            seed=0,
        )


# ---------------------------------------------------------------------------
# 6. Pipeline smoke per mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ("bilinear", "bicubic", "nearest-exact"))
def test_pipeline_runs_end_to_end_per_mode_with_correct_shapes_and_audio(mode):
    sigmas = torch.linspace(1.0, 0.0, 13)
    cfg = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(4,),
        transition_mode="explicit",
        noise_policy="direct_coarse",
        upscale_type=mode,
    )
    video_audible = 0.25
    x0_output = {}

    class AudioGuider:
        """Echo video guider whose audio carries a fixed non-zero level."""

        class Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

            def process_latent_out(self, x):
                return x

        def __init__(self):
            self.model_patcher = type("P", (), {"model": self.Model()})()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            pub_video, pub_audio = list(noise.unbind())
            out_audio = torch.full_like(pub_audio, video_audible)
            out = make_nested(pub_video, out_audio)
            if callback is not None:
                count = len(sigmas) - 1
                for i in range(count):
                    callback(i, out, out, count)
            return out

    out, _ = run_speed_pipeline(
        SeededRandomNoise(seed=5),
        AudioGuider(),
        sigmas,
        make_latent(h=8, w=8, t=2),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
        x0_output=x0_output,
    )
    video, audio = out["samples"].unbind()
    assert tuple(video.shape[-3:]) == (2, 8, 8)
    assert audio.shape == (1, 1, 2, 44)
    assert torch.count_nonzero(audio) > 0


def test_audio_output_is_unaffected_by_upscale_type_choice():
    """Identical setups except upscale_type give the same audio stream."""
    sigmas = torch.tensor([1.0, .8, .6, .4, .2, 0.0])
    audio_out = 1.3

    class AudioGuider:
        class Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

            def process_latent_out(self, x):
                return x

        def __init__(self):
            self.model_patcher = type("P", (), {"model": self.Model()})()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            pub_video, pub_audio = list(noise.unbind())
            out = make_nested(pub_video, torch.full_like(pub_audio, audio_out))
            if callback is not None:
                count = len(sigmas) - 1
                for i in range(count):
                    callback(i, out, out, count)
            return out

    runs = {}
    for mode in ("dct", "bilinear", "bicubic", "nearest-exact"):
        cfg = SpeedConfig(
            scales=(.5, 1.0),
            transition_steps=(2,),
            transition_mode="explicit",
            audio_policy="clock_reindex",
            upscale_type=mode,
        )
        noise = SeededRandomNoise(seed=42)
        x0_output = {}
        out, _ = run_speed_pipeline(
            noise,
            AudioGuider(),
            sigmas,
            make_latent(h=8, w=8, t=2),
            cfg,
            sampler_override=object(),
            disable_pbar=True,
            x0_output=x0_output,
        )
        video, audio = out["samples"].unbind()
        runs[mode] = (video, audio)
        assert torch.count_nonzero(audio) > 0

    dct_video, dct_audio = runs["dct"]
    for mode in ("bilinear", "bicubic", "nearest-exact"):
        _, mode_audio = runs[mode]
        assert torch.equal(mode_audio, dct_audio), f"audio differs for {mode}"


def test_non_dct_pipeline_smoke_with_fake_noise_and_recording_guider():
    """Non-dct path runs on the plain fake-noise suite harness too."""
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
        upscale_type="bicubic",
    )
    x0_output = {}
    out, _ = run_speed_pipeline(
        make_fake_noise(),
        make_recording_guider(),
        SIGMAS,
        make_latent(),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
        x0_output=x0_output,
    )
    video, _ = out["samples"].unbind()
    assert tuple(video.shape[-3:]) == (2, 8, 8)


# ---------------------------------------------------------------------------
# 7. Missing x0 fails closed
# ---------------------------------------------------------------------------

def test_non_dct_mode_without_x0_callback_raises_runtime_error():
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
        upscale_type="bilinear",
    )

    class NoCallbackGuider:
        """Guider whose callback never fires — x0_output stays empty."""

        class Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0

            def process_latent_out(self, x):
                return x

        def __init__(self):
            self.model_patcher = type("P", (), {"model": self.Model()})()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            return latent_image

    with pytest.raises(RuntimeError, match="requires an x0 callback"):
        run_speed_pipeline(
            make_fake_noise(),
            NoCallbackGuider(),
            SIGMAS,
            make_latent(),
            cfg,
            sampler_override=object(),
            disable_pbar=True,
        )


def test_dct_mode_without_x0_callback_still_works():
    """The legacy dct path must not require the preview callback."""
    cfg = SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
        upscale_type="dct",
    )
    out, _ = run_speed_pipeline(
        make_fake_noise(),
        make_recording_guider(callback_every_step=False),
        SIGMAS,
        make_latent(),
        cfg,
        sampler_override=object(),
        disable_pbar=True,
    )
    video, _ = out["samples"].unbind()
    assert tuple(video.shape[-3:]) == (2, 8, 8)
