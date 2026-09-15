"""Focused runtime coverage for RES boundary-history selection.

The existing V1 runtime suites intentionally pin the default ``reset`` path.
These tests exercise ``projected`` through the production
``run_speed_pipeline -> create_speed_sampler_handle -> on_transition`` chain,
including both the arithmetic nested test seam and the host-style flat packed
sampler seam.
"""

import math

import pytest
import torch

import speed_scripts.h3_runtime as h3_runtime
from conftest import SeededRandomNoise, make_latent, make_nested
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.sampler_support import create_speed_sampler_handle


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def _cfg():
    return SpeedConfig(
        scales=(.5, 1.0),
        transition_steps=(4,),
        transition_mode="explicit",
        audio_policy="carry_preserve",
    )


class ComputedNested:
    """Arithmetic-capable nested video/audio stand-in for direct sampler calls."""

    is_nested = True

    def __init__(self, streams):
        self.streams = list(streams)

    def unbind(self):
        return list(self.streams)

    def _pair(self, other, fn):
        if isinstance(other, ComputedNested):
            parts = other.streams
        else:
            parts = [other] * len(self.streams)
        return ComputedNested([fn(a, b) for a, b in zip(self.streams, parts)])

    def __add__(self, other):
        return self._pair(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._pair(other, lambda a, b: b + a)

    def __sub__(self, other):
        return self._pair(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._pair(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._pair(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self._pair(other, lambda a, b: b * a)

    def __truediv__(self, other):
        return self._pair(other, lambda a, b: a / b)


class _ModelInfo:
    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0

    def process_latent_out(self, x):
        return x


class NestedExecGuider:
    """Executes the actual stateful RES sampler on a nested test representation."""

    def __init__(self):
        self.model_patcher = type("P", (), {"model": _ModelInfo()})()
        self.stage_entries = []
        self.model_evals = 0

    def _model(self):
        guider = self

        class Model:
            def __call__(self, x, sigma, **kwargs):
                guider.model_evals += 1
                s = float(sigma)
                video, audio = x.unbind()
                return ComputedNested([
                    video * .7 + .3 * (s / (1.0 + s)),
                    audio * .8 + .02 * (s / (1.0 + s)),
                ])

        return Model()

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        state = getattr(sampler, "state", None)
        if state is None or state.old_denoised is None:
            self.stage_entries.append(None)
        else:
            video, audio = state.old_denoised.unbind()
            self.stage_entries.append({
                "video_shape": tuple(video.shape),
                "audio": audio.clone(),
                "old_sigma_down": state.old_sigma_down,
                "prev_sigma_in": state.prev_sigma_in,
            })

        run_noise = ComputedNested(list(noise.unbind()))
        count = len(sigmas) - 1

        adapted = None
        if callback is not None:
            def adapted(entry):
                callback(entry["i"], entry["denoised"], entry["x"], count)

        out = sampler(
            self._model(), run_noise, sigmas,
            extra_args=None, callback=adapted, disable=True,
        )
        return make_nested(*out.unbind())


def _capture_factory(monkeypatch):
    captured = []
    production_factory = create_speed_sampler_handle

    def factory(name, **kwargs):
        handle = production_factory(name, **kwargs)
        captured.append((name, kwargs, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    return captured


def test_projected_mode_runs_through_real_runtime_and_enters_next_stage_with_history(monkeypatch):
    captured = _capture_factory(monkeypatch)
    guider = NestedExecGuider()

    out, _ = run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), _cfg(),
        sampler_name="res_multistep",
        res_history_mode="projected",
        disable_pbar=True,
    )

    assert len(captured) == 1
    name, kwargs, handle = captured[0]
    assert name == "res_multistep"
    assert kwargs["res_history_mode"] == "projected"
    assert handle.history_mode == "projected"

    # Stage 0 starts with no history. After the 0.5 -> 1.0 boundary the
    # historical projected path supplies a target-geometry history record to
    # the final stage.
    assert guider.stage_entries[0] is None
    entry = guider.stage_entries[1]
    assert entry is not None
    assert entry["video_shape"][-3:] == (2, 8, 8)

    _, expected_down = aligned_sigma(.6, 2.0)
    _, expected_prev = aligned_sigma(.7, 2.0)
    assert entry["old_sigma_down"] == pytest.approx(expected_down)
    assert entry["prev_sigma_in"] == pytest.approx(expected_prev)

    video, audio = out["samples"].unbind()
    assert video.shape[-2:] == (8, 8)
    assert audio.ndim == 4
    assert guider.model_evals == len(SIGMAS) - 1

    # Runtime cleanup still owns the state even in projected mode.
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


def _pack_latents(streams):
    shapes = [tuple(t.shape) for t in streams]
    flat = [t.reshape(t.shape[0], 1, -1) for t in streams]
    return torch.cat(flat, dim=-1), shapes


def _unpack_latents(combined, shapes):
    out = []
    work = combined
    for shape in shapes:
        cut = math.prod(shape[1:])
        out.append(work[:, :, :cut].reshape([work.shape[0]] + list(shape)[1:]))
        work = work[:, :, cut:]
    return out


class HostFlatGuider:
    """Models CFGGuider's flat sampler seam while executing the real RES object."""

    def __init__(self):
        self.model_patcher = type("P", (), {"model": _ModelInfo()})()
        self.stage_entries = []
        self.stream_shapes = None
        self.model_evals = 0

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None):
        self.model_evals += 1
        s = float(sigma)
        video_elems = math.prod(self.stream_shapes[0][1:])
        video = x[:, :, :video_elems]
        audio = x[:, :, video_elems:]
        video = video * .7 + .3 * (s / (1.0 + s))
        audio = audio * .8 + .02 * (s / (1.0 + s))
        return torch.cat((video, audio), dim=-1)

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        nested_shapes = [tuple(t.shape) for t in latent_image.unbind()]
        self.stream_shapes = nested_shapes

        state = getattr(sampler, "state", None)
        if state is None or state.old_denoised is None:
            self.stage_entries.append(None)
        else:
            self.stage_entries.append({
                "shape": tuple(state.old_denoised.shape),
                "old_sigma_down": state.old_sigma_down,
                "prev_sigma_in": state.prev_sigma_in,
            })

        noise_flat, _ = _pack_latents(list(noise.unbind()))
        latent_flat, _ = _pack_latents(list(latent_image.unbind()))

        out_flat = sampler.sample(
            self,
            sigmas,
            {},
            callback,
            noise_flat,
            latent_image=latent_flat,
            denoise_mask=None,
            disable_pbar=True,
        )
        video, audio = _unpack_latents(out_flat, nested_shapes)
        return make_nested(video, audio)


def test_projected_mode_uses_runtime_source_shapes_on_flat_host_seam(monkeypatch):
    captured = _capture_factory(monkeypatch)
    guider = HostFlatGuider()
    latent = make_latent()
    full_video, full_audio = latent["samples"].unbind()

    out, _ = run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, latent, _cfg(),
        sampler_name="res_multistep",
        res_history_mode="projected",
        disable_pbar=True,
    )

    assert len(captured) == 1
    _, kwargs, handle = captured[0]
    assert kwargs["res_history_mode"] == "projected"
    assert guider.stage_entries[0] is None

    entry = guider.stage_entries[1]
    assert entry is not None
    assert entry["shape"] == (
        1,
        1,
        full_video[0].numel() + full_audio[0].numel(),
    )

    _, expected_down = aligned_sigma(.6, 2.0)
    _, expected_prev = aligned_sigma(.7, 2.0)
    assert entry["old_sigma_down"] == pytest.approx(expected_down)
    assert entry["prev_sigma_in"] == pytest.approx(expected_prev)

    video, audio = out["samples"].unbind()
    assert tuple(video.shape) == tuple(full_video.shape)
    assert tuple(audio.shape) == tuple(full_audio.shape)
    assert guider.model_evals == len(SIGMAS) - 1

    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None
