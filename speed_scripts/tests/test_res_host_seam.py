"""Host-seam tests for the RES Multistep adapter (plan S7 §13).

These tests exercise the two seams the real ComfyUI host builds and the
earlier runtime tests bypassed:

1. The sampler-OBJECT contract. The host ``CFGGuider.inner_sample`` wraps
   ``sampler.sample`` in its wrapper executor and calls
   ``executor.execute(self, sigmas, extra_args, callback, noise,
   latent_image, denoise_mask, disable_pbar)`` — the object must expose
   ``.sample`` with that shape. A guider here invokes the sampler object
   exactly that way.

2. The flat packed tensor at the sampler boundary. On a real nested-latent
   H3 run, ``CFGGuider.sample`` packs video+audio FLAT with
   ``comfy.utils.pack_latents`` (each stream reshaped ``[B, 1, -1]``,
   concatenated on the last axis) before any sampler code runs, and unpacks
   the sampler output back into a nested tensor afterwards. So the model's
   denoised output at the boundary — and therefore the RES history stored in
   ``ResMultistepState.old_denoised`` — is a plain flat tensor, and the
   ``on_transition`` history projection must split video from audio without
   crashing.

The conftest ``KSAMPLER`` stub models the host shape; these tests use it the
way ``CFGGuider`` uses the real one.
"""

import math

import pytest
import torch

import speed_scripts.h3_runtime as h3_runtime
from conftest import (
    LADDER_BOUNDARIES,
    SeededRandomNoise,
    make_latent,
)
from speed_scripts.automatic_config import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import _LW_ATTR, run_speed_pipeline
from speed_scripts.res_multistep_adapter import (
    ResMultistepSampler,
    ResMultistepState,
)
from speed_scripts.sampler_support import (
    SpeedTransition,
    create_res_multistep_sampler_handle,
)

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


# ---------------------------------------------------------------------------
# Host-shaped helpers
# ---------------------------------------------------------------------------

def _pack_latents(streams):
    """The host ``comfy.utils.pack_latents`` layout, verbatim."""
    shapes, tensors = [], []
    for tensor in streams:
        shapes.append(tuple(tensor.shape))
        tensors.append(tensor.reshape(tensor.shape[0], 1, -1))
    return torch.cat(tensors, dim=-1), shapes


def _unpack_latents(combined, shapes):
    """The host ``comfy.utils.unpack_latents`` layout, verbatim."""
    out, work = [], combined
    for shape in shapes:
        cut = math.prod(shape[1:])
        out.append(work[:, :, :cut].reshape([work.shape[0]] + list(shape)[1:]))
        work = work[:, :, cut:]
    return out


class HostShapedGuider:
    """A guider with the real host CFGGuider calling conventions.

    ``sample`` packs nested noise + latent flat, then invokes the sampler
    object through the host sampler-object contract: a WrapperExecutor-style
    ``getattr(sampler, "sample")`` lookup followed by the positional call
    ``sampler.sample(self, sigmas, extra_args, callback, noise, latent_image,
    denoise_mask, disable_pbar)``. The model callable is the host's
    ``KSamplerX0Inpaint`` shape (inner call receives only
    ``(x, sigma, model_options, seed)``). The sampler output is unpacked back
    into a nested pair, like ``CFGGuider.sample`` does. Records the sampler
    objects it saw and the state snapshot at each stage entry.

    Carries ``model_patcher.model`` with the sigma-shift attributes the
    SPEED runtime resolves from the guider, like the real host guider does.
    """

    class _Model:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

        def process_latent_out(self, x):
            return x

    def __init__(self):
        self.model_patcher = type("P", (), {"model": self._Model()})()
        self.samplers = []
        self.sigma_calls = []
        self.stage_entries = []
        self.model_evals = 0
        # Per-stream shapes of the most recent stage's nested latent (the
        # layout the model call must slice, set fresh by every sample() call).
        self.stream_shapes = [(1, 1, 2, 4, 4), (1, 1, 2, 8)]

    def __call__(self, x, sigma, denoise_mask=None, model_options=None, seed=None):
        # CFGGuider shape: the guider itself is the model wrapper KSAMPLER
        # wraps; its call produces the denoised prediction for the flat pack.
        # The prediction covers the same flat layout, so the video and audio
        # slices are transformed separately, like the real H3 model does.
        self.model_evals += 1
        s = float(sigma)
        video_elems = math.prod(self.stream_shapes[0][1:])
        video, audio = x[:, :, :video_elems], x[:, :, video_elems:]
        video = video * 0.7 + 0.3 * (s / (1.0 + s))
        audio = audio * 0.8 + 0.02 * (s / (1.0 + s))
        return torch.cat(
            (video.reshape(x.shape[0], 1, -1), audio.reshape(x.shape[0], 1, -1)),
            dim=-1,
        )

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        self.sigma_calls.append([float(s) for s in sigmas])
        nested_streams = list(latent_image.unbind())
        self.stream_shapes = [tuple(t.shape) for t in nested_streams]
        noise_flat, _ = _pack_latents(list(noise.unbind()))
        latent_flat, _ = _pack_latents(nested_streams)
        state = getattr(sampler, "state", None)
        if state is not None and state.old_denoised is not None:
            self.stage_entries.append(_FlatSnapshot(state))
        else:
            self.stage_entries.append(None)
        count = len(sigmas) - 1

        def host_callback(step, x0, x, total_steps):
            if callback is not None:
                callback(step, x0, x, total_steps)

        # The exact host invocation: sampler.sample(guider, sigmas,
        # extra_args, callback, noise, latent_image, denoise_mask, pbar).
        out_flat = sampler.sample(
            self, sigmas, {}, host_callback, noise_flat,
            latent_image=latent_flat, denoise_mask=None, disable_pbar=True,
        )
        video, audio = _unpack_latents(
            out_flat, [tuple(t.shape) for t in latent_image.unbind()]
        )
        from conftest import make_nested
        return make_nested(video, audio)


class _FlatSnapshot:
    """What the run-scoped RES state held when one stage began (flat path)."""

    def __init__(self, state):
        self.old_denoised = state.old_denoised
        self.old_sigma_down = state.old_sigma_down
        self.prev_sigma_in = state.prev_sigma_in


def _flat_ladder_cfg(stages, **overrides):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
        audio_policy="carry_preserve",
        **overrides,
    )


def _run_flat_host(cfg, guider, monkeypatch):
    handle = create_res_multistep_sampler_handle()
    captured = []

    def factory(name):
        captured.append((name, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    out, denoised = run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name="res_multistep", disable_pbar=True,
    )
    assert captured == [("res_multistep", handle)]
    return handle, out, denoised


# ---------------------------------------------------------------------------
# Critical 1: the host sampler-object contract (.sample)
# ---------------------------------------------------------------------------

def test_res_sampler_object_exposes_host_sample_contract():
    """The attribute the host resolves is ``.sample`` (WrapperExecutor wraps
    ``sampler.sample``), with the host's positional shape."""
    sampler = ResMultistepSampler()
    assert hasattr(sampler, "sample")
    import inspect
    params = [p for p in inspect.signature(sampler.sample).parameters if p != "self"]
    assert params == [
        "model_wrap", "sigmas", "extra_args", "callback", "noise",
        "latent_image", "denoise_mask", "disable_pbar",
    ]


def test_host_shaped_sample_call_runs_and_preserves_state():
    """A host-shaped ``.sample`` call runs the real stateful RES function and
    routes through the same state object a follow-up call sees. The guider
    doubles as the model callable (``KSamplerX0Inpaint`` wraps the guider's
    model, which the wrapper invokes with only ``(x, sigma, ...)``)."""
    sampler = ResMultistepSampler()
    seen = []

    def model(x, sigma, denoise_mask=None, model_options=None, seed=None):
        seen.append(sigma)
        return x * 0.7

    noise_video = torch.randn(1, 1, 2, 4, 4)
    noise_audio = torch.randn(1, 1, 2, 8)
    noise_flat, _ = _pack_latents([noise_video, noise_audio])
    latent_flat, _ = _pack_latents([
        torch.zeros_like(noise_video), torch.zeros_like(noise_audio)
    ])
    out = sampler.sample(
        model, SIGMAS[:5], {}, None, noise_flat,
        latent_image=latent_flat, denoise_mask=None, disable_pbar=True,
    )
    assert out.shape == noise_flat.shape
    # Four intervals, each evaluated once, with the denoise_mask contract the
    # host KSAMPLER injects into extra_args.
    assert len(seen) == 4
    # History was written by the run and is the host's flat tensor shape.
    assert sampler.state.old_denoised is not None
    assert torch.is_tensor(sampler.state.old_denoised)
    assert sampler.state.old_denoised.shape == noise_flat.shape
    assert sampler.state.old_sigma_down == pytest.approx(0.6)
    assert sampler.state.prev_sigma_in == pytest.approx(0.7)
    # A second host-shaped call carries the same state forward.
    out2 = sampler.sample(
        model, SIGMAS[4:], {}, None, noise_flat,
        latent_image=latent_flat, denoise_mask=None, disable_pbar=True,
    )
    assert out2.shape == out.shape
    # 4 intervals in the first call, 6 in the second.
    assert len(seen) == 10


def test_res_multistep_state_clear_releases_history():
    state = ResMultistepState()
    state.old_denoised = torch.zeros(1, 1, 4)
    state.old_sigma_down = 0.5
    state.prev_sigma_in = 0.7
    state.clear()
    assert state.old_denoised is None
    assert state.old_sigma_down is None
    assert state.prev_sigma_in is None


# ---------------------------------------------------------------------------
# Critical 2: flat packed history at the sampler boundary
# ---------------------------------------------------------------------------

def test_flat_history_on_transition_projects_video_and_preserves_audio():
    """A flat packed history (the only shape the real host produces at the
    sampler boundary) is projected without crashing: video geometry advances,
    audio elements pass through bit-exact, and the result re-packs flat."""
    video = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 2, 4) / 16
    audio = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) / 8
    flat, shapes = _pack_latents([video, audio])
    handle = create_res_multistep_sampler_handle()
    handle.state.old_denoised = flat
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.7
    handle.on_transition(SpeedTransition(
        stage_idx=0, ratio=2.0, old_sigma=0.5, new_sigma=0.4,
        source_thw=(2, 2, 4), target_thw=(2, 4, 8),
        source_stream_shapes=shapes,
    ))
    projected = handle.state.old_denoised
    assert torch.is_tensor(projected)
    # Target geometry (1, 1, 2, 4, 8) = 64 video elems + 8 audio elems.
    assert projected.shape == (1, 1, 72)
    p_video, p_audio = _unpack_latents(
        projected, [(1, 1, 2, 4, 8), (1, 1, 2, 4)]
    )
    assert tuple(p_video.shape[-3:]) == (2, 4, 8)
    # Audio is preserved bit-exact through the projection.
    assert torch.equal(p_audio, audio)
    # Sigma metadata was rebased onto the aligned coordinates.
    assert handle.state.old_sigma_down == pytest.approx(0.4)


def test_flat_history_on_transition_requires_stream_shapes():
    """A flat history without the pack's shapes fails closed instead of
    guessing a slice boundary."""
    handle = create_res_multistep_sampler_handle()
    handle.state.old_denoised = torch.zeros(1, 1, 40)
    with pytest.raises(ValueError, match="per-stream shapes"):
        handle.on_transition(SpeedTransition(
            stage_idx=0, ratio=2.0, old_sigma=0.5, new_sigma=0.4,
            source_thw=(2, 2, 4), target_thw=(2, 4, 4),
        ))


# ---------------------------------------------------------------------------
# End-to-end: a CFGGuider-shaped guider through the real runtime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3))
def test_runtime_with_cfgguider_shaped_guider_completes(monkeypatch, stages):
    """Full run with the guider shaped like the real host ``CFGGuider``:
    nested in the runtime's hands, flat at the sampler boundary, sampler
    invoked via ``.sample``. Both criticals crashed exactly here before the
    fix — no ``.sample`` attribute, then ValueError at the first transition."""
    guider = HostShapedGuider()
    handle, out, denoised = _run_flat_host(_flat_ladder_cfg(stages), guider, monkeypatch)

    assert len(guider.sigma_calls) == stages
    assert guider.model_evals == len(SIGMAS) - 1
    # Every stage received the SAME run-scoped sampler object.
    assert guider.samplers == [guider.samplers[0]] * stages
    assert isinstance(guider.samplers[0], ResMultistepSampler)
    video, audio = out["samples"].unbind()
    assert tuple(video.shape[-2:]) == (8, 8)
    assert audio.ndim == 4
    # Stage entries: stage 0 empty; later stages carry flat packed history.
    assert guider.stage_entries[0] is None
    for snap in guider.stage_entries[1:]:
        assert snap is not None
        assert torch.is_tensor(snap.old_denoised)
        assert snap.old_denoised.ndim == 3  # the host's flat pack
    # Run-level cleanup ran.
    assert not hasattr(guider, _LW_ATTR)
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None
