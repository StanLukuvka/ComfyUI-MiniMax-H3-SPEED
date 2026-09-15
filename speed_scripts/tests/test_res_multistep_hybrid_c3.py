"""Commit 3 tests for the mechanical RES hybrid candidate blend."""


import pytest
import torch

from speed_scripts.res_multistep_adapter import (
    ResMultistepState,
    _res_first_order_update,
    _res_second_order_update,
    hybridize_res_candidates,
    res_multistep_sampler,
)
from speed_scripts.spectral import dct2, dct_temporal


class Nested:
    is_nested = True

    def __init__(self, streams):
        self.streams = list(streams)

    def unbind(self):
        return list(self.streams)


def _pack(video, audio):
    return torch.cat((video.reshape(video.shape[0], 1, -1), audio.reshape(audio.shape[0], 1, -1)), dim=-1)


def test_nested_hybrid_replaces_only_video_source_block_and_keeps_audio_first():
    first_video = torch.randn(1, 1, 3, 4, 6, dtype=torch.float64)
    second_video = torch.randn_like(first_video)
    first_audio = torch.randn(1, 1, 2, 5, dtype=torch.float64)
    second_audio = torch.randn_like(first_audio)
    out = hybridize_res_candidates(
        Nested([first_video, first_audio]), Nested([second_video, second_audio]),
        (2, 3, 4), None,
    )
    video, audio = out.unbind()
    expected_coeff = dct2(dct_temporal(first_video))
    expected_coeff[..., :2, :3, :4] = dct2(dct_temporal(second_video))[..., :2, :3, :4]
    assert video.dtype == first_video.dtype
    assert torch.allclose(dct2(dct_temporal(video)), expected_coeff, atol=1e-5, rtol=0)
    assert torch.equal(audio, first_audio)


def test_flat_hybrid_requires_explicit_matching_stream_shapes_and_repacks():
    video_1 = torch.randn(1, 1, 3, 4, 4)
    audio_1 = torch.randn(1, 1, 2, 5)
    video_2 = torch.randn_like(video_1)
    audio_2 = torch.randn_like(audio_1)
    first = _pack(video_1, audio_1)
    second = _pack(video_2, audio_2)
    shapes = (tuple(video_1.shape), tuple(audio_1.shape))
    out = hybridize_res_candidates(first, second, (2, 3, 3), shapes)
    assert out.shape == first.shape
    assert torch.equal(out[..., video_1.numel():], audio_1.reshape(1, 1, -1))
    with pytest.raises(ValueError, match="target stream shapes"):
        hybridize_res_candidates(first, second, (2, 3, 3), None)


def test_hybrid_sampler_blends_candidates_after_one_model_call():
    video_shape = (1, 1, 2, 2, 2)
    audio_shape = (1, 1, 1, 2)
    shapes = (video_shape, audio_shape)
    x = torch.arange(10, dtype=torch.float32).reshape(1, 1, -1)
    old = x + 1
    state = ResMultistepState(
        old_denoised=old,
        old_sigma_down=0.9,
        prev_sigma_in=1.0,
        hybrid_pending=True,
        hybrid_second_order_thw=(1, 1, 1),
        hybrid_target_stream_shapes=shapes,
    )
    calls = 0

    def model(value, sigma, **kwargs):
        nonlocal calls
        calls += 1
        return value * 0.25

    sigmas = torch.tensor([0.8, 0.6])
    out = res_multistep_sampler(model, x, sigmas, state)
    denoised = x * 0.25
    first = _res_first_order_update(x, denoised, 0.8, 0.6)
    second = _res_second_order_update(x, denoised, old, 0.8, 0.6, 0.9, 1.0)
    expected = hybridize_res_candidates(first, second, (1, 1, 1), shapes)
    assert calls == 1
    assert torch.allclose(out, expected)
    assert state.hybrid_pending is False


def test_hybrid_rejects_source_geometry_that_exceeds_one_target_axis():
    video = torch.zeros(1, 1, 3, 4, 6)
    audio = torch.zeros(1, 1, 1, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        hybridize_res_candidates(
            Nested([video, audio]), Nested([video, audio]), (2, 5, 4), None,
        )


def test_hybrid_pending_is_consumed_when_second_order_is_unavailable():
    state = ResMultistepState(
        hybrid_pending=True,
        hybrid_second_order_thw=(1, 1, 1),
        hybrid_target_stream_shapes=((1, 1, 1, 1, 1), (1, 1, 1, 1)),
    )
    model_calls = 0

    def model(x, sigma, **kwargs):
        nonlocal model_calls
        model_calls += 1
        return x

    x = torch.ones(1, 1, 2, 2, 2)
    out = res_multistep_sampler(model, x, torch.tensor([1.0, 0.0]), state)
    assert model_calls == 1
    assert torch.equal(out, x)
    assert state.hybrid_pending is False
    assert state.hybrid_second_order_thw is None
    assert state.hybrid_target_stream_shapes is None
