"""RES Multistep through the real SPEED runtime (plan S7 §20-§24, §31 steps 42-46).

B1/B2 pinned the adapter and the transition hook in isolation; this module
runs ``res_multistep`` through ``run_speed_pipeline`` — the real stage loop,
the real transition call site, and the real cleanup chain — through the
public ``res_multistep`` selector and the runtime's handle factory.

The conftest ``NestedTensor`` stub that ``h3_runtime.pack_latent`` builds has
no tensor operations, so the guider below unwraps stage tensors into an
arithmetic-capable nested pair before the sampler runs and wraps the result
back — mirroring the real host nested-tensor contract the adapter is written
against. The guider executes the sampler object the runtime hands it, so
every interval runs the real stateful RES adapter and the carried
``ResMultistepState`` can be inspected at each stage entry.
"""


import pytest
import torch

import speed_scripts.h3_runtime as h3_runtime
import speed_scripts.res_multistep_adapter as res_multistep_adapter
from conftest import (
    LADDER_BOUNDARIES,
    RecordingEchoGuider,
    SeededRandomNoise,
    make_latent,
    make_nested,
)
from speed_scripts.automatic_config import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.h3_runtime import _LW_ATTR, run_speed_pipeline
from speed_scripts.res_multistep_adapter import (
    ResMultistepSampler,
)
from speed_scripts.sampler_support import create_speed_sampler_handle

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


class ComputedNested:
    """Arithmetic-capable nested H3 stand-in (video [B,C,T,H,W] + audio).

    Same unbind/is_nested contract as the host nested tensor, plus the
    tensor operations the RES solver applies to the working state
    (add/sub/mul/div against floats and 0-dim schedule tensors).
    """

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


class _Snapshot:
    """What the run-scoped RES state held when one stage began."""

    def __init__(self, state):
        video, audio = state.old_denoised.unbind()
        self.video = video
        self.audio = audio
        self.old_sigma_down = state.old_sigma_down
        self.prev_sigma_in = state.prev_sigma_in


class ResExecGuider(RecordingEchoGuider):
    """Echo guider that actually executes the sampler object it receives.

    ``RecordingEchoGuider`` never calls the sampler, so a stateful RES run
    would look like a no-op. This guider hands the received sampler object
    (the runtime handle's real ``ResMultistepSampler``) the stage noise and
    zero latent as arithmetic nested pairs, records one state snapshot per
    stage entry, counts model evaluations, keeps every denoised estimate the
    model produced, and wraps the sampler output back into the stub nested
    container the runtime unpacks.
    """

    def __init__(self):
        super().__init__()
        self.model_evals = 0
        self.model_outputs = []
        self.stage_entries = []

    def _model(self):
        guider = self

        class _Model:
            def __call__(self, x, sigma, **extra):
                guider.model_evals += 1
                s = float(sigma)
                video, audio = x.unbind()
                denoised = ComputedNested([
                    video * 0.7 + 0.3 * (s / (1.0 + s)),
                    audio * 0.8 + 0.02 * (s / (1.0 + s)),
                ])
                guider.model_outputs.append(denoised)
                return denoised

        return _Model()

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        self.sigma_calls.append([float(s) for s in sigmas])
        pub_video, pub_audio = list(noise.unbind())
        self.noise_shapes.append(tuple(pub_video.shape))
        state = getattr(sampler, "state", None)
        if state is not None and state.old_denoised is not None:
            self.stage_entries.append(_Snapshot(state))
        else:
            self.stage_entries.append(None)
        run_noise = ComputedNested([pub_video, pub_audio])
        count = len(sigmas) - 1

        adapted = None
        if callback is not None:
            def adapted(entry):
                callback(entry["i"], entry["denoised"], entry["x"], count)
        out = sampler(
            self._model(), run_noise, sigmas,
            extra_args=None, callback=adapted, disable=True,
        )
        out_video, out_audio = out.unbind()
        return make_nested(out_video, out_audio)


def _capture_public_factory(monkeypatch):
    """Wrap the public factory without replacing its production behavior."""
    captured = []

    def factory(name, **kwargs):
        handle = create_speed_sampler_handle(name, **kwargs)
        captured.append((name, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    return captured


def _assert_res_seam(captured, handle):
    assert captured == [("res_multistep", handle)]


def _run_res(cfg, guider, monkeypatch, **kwargs):
    captured = _capture_public_factory(monkeypatch)
    out, denoised = run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name="res_multistep", disable_pbar=True, **kwargs,
    )
    handle = captured[0][1]
    _assert_res_seam(captured, handle)
    return handle, out, denoised


def _explicit_ladder_cfg(stages, **overrides):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
        **overrides,
    )


def _automatic_calibrated_cfg(stages):
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


def _assert_res_state_cleared(handle):
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_stage_ladder_completes_at_full_resolution(monkeypatch, stages):
    guider = ResExecGuider()
    handle, out, denoised = _run_res(_explicit_ladder_cfg(stages), guider, monkeypatch)
    assert len(guider.sigma_calls) == stages
    assert guider.model_evals == len(SIGMAS) - 1
    assert guider.samplers == [guider.samplers[0]] * stages
    assert isinstance(guider.samplers[0], ResMultistepSampler)
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    assert not hasattr(guider, _LW_ATTR)
    _assert_res_state_cleared(handle)


@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_callback_count_equals_global_denoising_intervals(monkeypatch, stages):
    seen = []
    guider = ResExecGuider()
    _run_res(
        _explicit_ladder_cfg(stages), guider, monkeypatch,
        preview_callback=lambda step, x0, x, total: seen.append((step, total)),
    )
    assert seen == [(i, 10) for i in range(10)]


@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_reset_mode_starts_each_stage_without_history(monkeypatch, stages):
    guider = ResExecGuider()
    _run_res(_explicit_ladder_cfg(stages), guider, monkeypatch)
    entries = guider.stage_entries
    assert entries[0] is None
    assert entries[1:] == [None] * (stages - 1)


@pytest.mark.parametrize("noise_policy", ("direct_coarse", "coupled_full_grid"))
def test_res_runtime_completes_under_noise_policy(monkeypatch, noise_policy):
    guider = ResExecGuider()
    handle, out, denoised = _run_res(
        _explicit_ladder_cfg(2, noise_policy=noise_policy), guider, monkeypatch,
    )
    assert len(guider.sigma_calls) == 2
    assert guider.model_evals == len(SIGMAS) - 1
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    assert guider.stage_entries[1] is None
    _assert_res_state_cleared(handle)


def test_res_noise_policies_produce_different_runs(monkeypatch):
    outs = []
    for policy in ("direct_coarse", "coupled_full_grid"):
        guider = ResExecGuider()
        _, out, _ = _run_res(
            _explicit_ladder_cfg(2, noise_policy=policy), guider, monkeypatch,
        )
        video, _ = out["samples"].unbind()
        outs.append(video)
    assert not torch.equal(outs[0], outs[1])


def _i2v_guider(guider_cls=ResExecGuider):
    guider = guider_cls()
    g = torch.Generator().manual_seed(5)
    keyframes = [
        {"latent": torch.randn(1, 1, 2, 8, 8, generator=g)} for _ in range(2)
    ]
    refs = [{"latent": torch.randn(1, 1, 2, 8, 8, generator=g)} for _ in range(2)]
    pristine = [kf["latent"].clone() for kf in keyframes]
    guider.original_conds = {
        "positive": [{
            "minimax_keyframes": keyframes,
            "minimax_refs": refs,
        }],
        "negative": [],
    }
    return guider, keyframes, refs, pristine


def _assert_pristine_conds(keyframes, refs, pristine):
    for kf, saved in zip(keyframes, pristine):
        assert torch.equal(kf["latent"], saved)
    for ref in refs:
        assert tuple(ref["latent"].shape[-2:]) == (8, 8)


def test_res_i2v_smoke_restores_pristine_and_keeps_history_separate(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    _, out, denoised = _run_res(_explicit_ladder_cfg(3), guider, monkeypatch)
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    _assert_pristine_conds(keyframes, refs, pristine)
    assert not hasattr(guider, _LW_ATTR)
    assert guider.stage_entries[-1] is None


class ExplodingAtStage2(ResExecGuider):
    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        if len(self.sigma_calls) == 2:
            raise RuntimeError("sampler call exploded")
        return super().sample(
            noise, latent_image, sampler, sigmas, callback=callback, **kwargs
        )


def test_failure_during_sampler_call_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider(ExplodingAtStage2)
    captured = _capture_public_factory(monkeypatch)
    with pytest.raises(RuntimeError, match="sampler call exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(3),
            sampler_name="res_multistep", disable_pbar=True,
        )
    handle = captured[0][1]
    _assert_res_seam(captured, handle)
    assert len(guider.stage_entries) == 2
    assert guider.stage_entries[1] is None
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_failure_during_spectral_transition_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    captured = _capture_public_factory(monkeypatch)
    observed = {}

    def exploding_expand(value, target_hw, sigma, seed):
        observed["history_existed"] = captured[0][1].state.old_denoised is not None
        raise RuntimeError("spectral transition exploded")

    monkeypatch.setattr(h3_runtime, "spectral_expand", exploding_expand)
    with pytest.raises(RuntimeError, match="spectral transition exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(2),
            sampler_name="res_multistep", disable_pbar=True,
        )
    handle = captured[0][1]
    _assert_res_seam(captured, handle)
    assert observed["history_existed"] is True
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_failure_during_audio_transition_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    captured = _capture_public_factory(monkeypatch)
    observed = {}

    def exploding_reindex(*args, **kwargs):
        observed["history_existed"] = captured[0][1].state.old_denoised is not None
        raise RuntimeError("audio transition exploded")

    monkeypatch.setattr(h3_runtime, "clock_reindex_audio_state", exploding_reindex)
    with pytest.raises(RuntimeError, match="audio transition exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(2),
            sampler_name="res_multistep", disable_pbar=True,
        )
    handle = captured[0][1]
    _assert_res_seam(captured, handle)
    assert observed["history_existed"] is True
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_res_second_generation_after_completed_run_starts_clean(monkeypatch):
    cfg = _explicit_ladder_cfg(3)
    guider = ResExecGuider()
    handle_1, out_1, _ = _run_res(cfg, guider, monkeypatch)
    _assert_res_state_cleared(handle_1)
    handle_2, out_2, _ = _run_res(cfg, guider, monkeypatch)
    assert handle_2 is not handle_1
    assert guider.stage_entries[3] is None
    video_1, audio_1 = out_1["samples"].unbind()
    video_2, audio_2 = out_2["samples"].unbind()
    assert torch.equal(video_1, video_2)
    assert torch.equal(audio_1, audio_2)
    control = ResExecGuider()
    _, out_control, _ = _run_res(cfg, control, monkeypatch)
    video_c, audio_c = out_control["samples"].unbind()
    assert torch.equal(video_2, video_c)
    assert torch.equal(audio_2, audio_c)
    assert not hasattr(guider, _LW_ATTR)


def test_res_coincident_boundary_zero_step_stages_remain_empty(monkeypatch):
    guider = ResExecGuider()
    _, out, _ = _run_res(_automatic_calibrated_cfg(3), guider, monkeypatch)
    assert len(guider.sigma_calls) == 3
    assert len(guider.sigma_calls[1]) == 1
    assert guider.model_evals == len(SIGMAS) - 1
    entries = guider.stage_entries
    assert entries[1] is None
    assert entries[2] is None
    _assert_full_res_nested(out)
    assert not hasattr(guider, _LW_ATTR)


def test_res_reset_mode_matches_explicit_boundary_reset_control(monkeypatch):
    cfg = _explicit_ladder_cfg(2)

    class ResetAtBoundary(ResExecGuider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if len(self.sigma_calls) == 1:
                sampler.state.clear()
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    _, out_default, _ = _run_res(cfg, ResExecGuider(), monkeypatch)
    handle_reset, out_reset, _ = _run_res(cfg, ResetAtBoundary(), monkeypatch)
    assert handle_reset.state.old_denoised is None
    assert torch.equal(out_default["samples"].unbind()[0], out_reset["samples"].unbind()[0])
