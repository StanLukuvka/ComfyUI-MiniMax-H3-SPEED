"""RES Multistep through the real SPEED runtime (plan S7 §20-§24, §31 steps 42-46).

B1/B2 pinned the adapter and the transition hook in isolation; this module
runs ``res_multistep`` through ``run_speed_pipeline`` — the real stage loop,
the real transition call site, and the real cleanup chain — with the §13
run-scoped handle injected at the runtime's handle factory (the same seam
the S6 lifecycle tests use; ``res_multistep`` joins the public selector only
at plan step 50).

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
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import _LW_ATTR, run_speed_pipeline
from speed_scripts.res_multistep_adapter import (
    ResMultistepSampler,
    project_clean_history,
)
from speed_scripts.sampler_support import create_res_multistep_sampler_handle

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
        x = ComputedNested(list(latent_image.unbind()))
        run_noise = ComputedNested([pub_video, pub_audio])
        count = len(sigmas) - 1

        adapted = None
        if callback is not None:
            def adapted(entry):
                # One native per-interval dict -> one legacy 4-arg callback,
                # the same adaptation every native sampler callback gets.
                callback(entry["i"], entry["denoised"], entry["x"], count)
        out = sampler(
            self._model(), run_noise, sigmas,
            extra_args=None, callback=adapted, disable=True,
        )
        out_video, out_audio = out.unbind()
        return make_nested(out_video, out_audio)


def _inject_res_handle(monkeypatch):
    """Route the runtime's handle factory to one fresh run-scoped RES handle.

    The §13 seam: the stage loop only ever talks to a ``SpeedSamplerHandle``.
    Until plan step 50 promotes ``res_multistep`` into the public selector,
    the tests inject the handle the §13 factory builds — the exact object
    production will build. The factory records the sampler name each run
    requests, so every test can prove the runtime actually asked for
    ``res_multistep`` instead of silently receiving a patched-in handle.
    Returns ``(handle, captured)``; ``captured`` accumulates one
    ``(name, handle)`` entry per run built on this patch.
    """
    handle = create_res_multistep_sampler_handle()
    captured = []

    def factory(name):
        captured.append((name, handle))
        return handle

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", factory)
    return handle, captured


def _assert_res_seam(captured, handle):
    """The runtime requested ``res_multistep`` and got this run-scoped handle."""
    assert captured == [("res_multistep", handle)]


def _run_res(cfg, guider, monkeypatch, **kwargs):
    handle, captured = _inject_res_handle(monkeypatch)
    out, denoised = run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name="res_multistep", disable_pbar=True, **kwargs,
    )
    _assert_res_seam(captured, handle)  # one run-scoped handle per run
    return handle, out, denoised


def _explicit_ladder_cfg(stages, **overrides):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
        **overrides,
    )


def _automatic_calibrated_cfg(stages):
    """The Automatic node's delta_custom config: both transitions quantize
    onto schedule index 1, so the middle stage gets a single-sigma schedule
    (zero denoising steps) — the legal coincident-boundary case."""
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=tuple(range(1, len(STAGES_TO_SCALES[stages]))),
        transition_mode="delta_custom",
        delta=.005,
        noise_amplitude=12.105,
        noise_decay_exponent=.773,
        full_latent_h=8,
        full_latent_w=8,
    )


def _interval_counts(stages):
    bounds = LADDER_BOUNDARIES[stages]
    edges = (0,) + bounds + (len(SIGMAS) - 1,)
    return [b - a for a, b in zip(edges[:-1], edges[1:])]


def _assert_full_res_nested(latent):
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


def _stage_geometry(stages, stage_idx):
    """(t, h, w) the ladder runs stage ``stage_idx`` at (no temporal scaling)."""
    scale = STAGES_TO_SCALES[stages][stage_idx]
    return (2, max(1, round(8 * scale)), max(1, round(8 * scale)))


def _assert_res_state_cleared(handle):
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


# ---------------------------------------------------------------------------
# §20 runtime completion — 2/3/4-stage ladders through the real adapter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_stage_ladder_completes_at_full_resolution(monkeypatch, stages):
    guider = ResExecGuider()
    handle, out, denoised = _run_res(_explicit_ladder_cfg(stages), guider, monkeypatch)

    assert len(guider.sigma_calls) == stages
    # Every global denoising interval ran through the real stateful adapter
    # (one model evaluation per interval).
    assert guider.model_evals == len(SIGMAS) - 1
    # Every stage received the SAME run-scoped sampler object.
    assert guider.samplers == [guider.samplers[0]] * stages
    assert isinstance(guider.samplers[0], ResMultistepSampler)
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    # Walker and RES handle cleanup both succeeded on the success path.
    assert not hasattr(guider, _LW_ATTR)
    _assert_res_state_cleared(handle)


# ---------------------------------------------------------------------------
# §22 RES progress — one callback per outer interval
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_callback_count_equals_global_denoising_intervals(monkeypatch, stages):
    seen = []
    guider = ResExecGuider()
    _run_res(
        _explicit_ladder_cfg(stages), guider, monkeypatch,
        preview_callback=lambda step, x0, x, total: seen.append((step, total)),
    )
    # The ladder tiles the 10-interval schedule; RES adds no callbacks of its
    # own, so the runtime sees exactly one per global interval.
    assert seen == [(i, 10) for i in range(10)]


# ---------------------------------------------------------------------------
# §20 state transport — configured transitions carried through state,
# alignment rebase exercised across stage boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stages", (2, 3, 4))
def test_res_state_is_carried_and_rebased_across_stage_boundaries(monkeypatch, stages):
    """At every stage boundary k the carried state must hold
    ``old_sigma_down == new_q`` and ``prev_sigma_in`` equal to the last
    interval's input sigma mapped through ``aligned_sigma`` with that
    boundary's ratio, with the video history already projected to the next
    stage's geometry. All expected values are computed independently from
    the schedule and the scale ladder."""
    guider = ResExecGuider()
    _run_res(_explicit_ladder_cfg(stages), guider, monkeypatch)

    bounds = LADDER_BOUNDARIES[stages]
    scales = STAGES_TO_SCALES[stages]
    entries = guider.stage_entries
    assert entries[0] is None  # a fresh run starts with empty history
    for k in range(1, stages):
        boundary = bounds[k - 1]
        ratio = scales[k] / scales[k - 1]
        new_q = aligned_sigma(float(SIGMAS[boundary]), ratio)[1]
        last_input = float(SIGMAS[boundary - 1])
        snap = entries[k]
        assert snap is not None
        assert snap.old_sigma_down == pytest.approx(new_q, rel=1e-6)
        assert snap.prev_sigma_in == pytest.approx(
            aligned_sigma(last_input, ratio)[1], rel=1e-6
        )
        # History video was projected to the next stage's geometry; the
        # clean audio estimate kept its shape and is nonzero.
        assert tuple(snap.video.shape[-3:]) == _stage_geometry(stages, k)
        assert tuple(snap.audio.shape) == (1, 1, 2, 44)
        assert snap.audio.abs().sum() > 0
        # §21: the clean history audio object is the model's own estimate —
        # passed through the boundary untouched, never clock-reindexed.
        intervals_before = sum(_interval_counts(stages)[:k])
        assert snap.audio is guider.model_outputs[intervals_before - 1].unbind()[1]


# ---------------------------------------------------------------------------
# §20 noise policies — direct_coarse and coupled_full_grid
# ---------------------------------------------------------------------------

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
    # The boundary rebase ran under this noise policy too.
    snap = guider.stage_entries[1]
    assert snap is not None
    boundary = LADDER_BOUNDARIES[2][0]  # the (0.5 -> 1.0) ladder's boundary
    ratio = STAGES_TO_SCALES[2][1] / STAGES_TO_SCALES[2][0]
    assert snap.old_sigma_down == pytest.approx(
        aligned_sigma(float(SIGMAS[boundary]), ratio)[1], rel=1e-6
    )
    assert snap.prev_sigma_in == pytest.approx(
        aligned_sigma(float(SIGMAS[boundary - 1]), ratio)[1], rel=1e-6
    )
    assert tuple(snap.video.shape[-3:]) == (2, 8, 8)
    _assert_res_state_cleared(handle)  # run-level cleanup ran


def test_res_noise_policies_produce_different_runs(monkeypatch):
    """The two policies must actually diverge (the coupled policy feeds the
    full-grid noise projection instead of fresh coarse noise), so passing
    one for the other cannot hide behind identical outputs."""
    outs = []
    for policy in ("direct_coarse", "coupled_full_grid"):
        guider = ResExecGuider()
        _, out, _ = _run_res(
            _explicit_ladder_cfg(2, noise_policy=policy), guider, monkeypatch,
        )
        video, _ = out["samples"].unbind()
        outs.append(video)
    assert not torch.equal(outs[0], outs[1])


# ---------------------------------------------------------------------------
# §24 RES I2V smoke — walker restore, pristine conds, history separate from
# conditioning
# ---------------------------------------------------------------------------

def _i2v_guider(guider_cls=ResExecGuider):
    """Guider with real I2V-shaped (nonzero) conditioning attached.

    Returns (guider, keyframes, refs, pristine) where pristine holds a clone
    of every keyframe latent for the restore assertions.
    """
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
    # Walker restore: keyframe latents came back to pristine full-res values,
    # refs were never resized, and no sampler code rebuilt the cond containers.
    _assert_pristine_conds(keyframes, refs, pristine)
    assert not hasattr(guider, _LW_ATTR)
    # §24: RES history projection is separate from I2V conditioning
    # projection — the final stage entered with projected solver history
    # that is no conditioning latent.
    snap = guider.stage_entries[-1]
    assert snap is not None
    assert snap.video.abs().sum() > 0
    for holder in keyframes + refs:
        assert snap.video is not holder["latent"]
        assert not torch.equal(snap.video, holder["latent"])


# ---------------------------------------------------------------------------
# §20 failure/cleanup — every failure point clears RES state, closes the
# handle, and still runs the walker cleanup through the nested finally chain
# ---------------------------------------------------------------------------

class ExplodingAtStage2(ResExecGuider):
    """Fails inside the third guider.sample call (the final stage)."""

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        if len(self.sigma_calls) == 2:
            raise RuntimeError("sampler call exploded")
        return super().sample(
            noise, latent_image, sampler, sigmas, callback=callback, **kwargs
        )


def test_failure_during_sampler_call_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider(ExplodingAtStage2)
    handle, captured = _inject_res_handle(monkeypatch)
    with pytest.raises(RuntimeError, match="sampler call exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(3),
            sampler_name="res_multistep", disable_pbar=True,
        )
    _assert_res_seam(captured, handle)
    # Two stages had run: RES history existed mid-run, so the cleared state
    # below proves the close, not an empty run.
    assert len(guider.stage_entries) == 2
    assert guider.stage_entries[1] is not None
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_failure_during_spectral_transition_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    handle, captured = _inject_res_handle(monkeypatch)
    observed = {}

    def exploding_expand(value, target_hw, sigma, seed):
        # Observed at the moment of explosion: stage 0 had completed real
        # intervals, so solver history existed when the transition failed.
        observed["history_existed"] = handle.state.old_denoised is not None
        raise RuntimeError("spectral transition exploded")

    monkeypatch.setattr(h3_runtime, "spectral_expand", exploding_expand)
    with pytest.raises(RuntimeError, match="spectral transition exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(2),
            sampler_name="res_multistep", disable_pbar=True,
        )
    _assert_res_seam(captured, handle)
    assert observed["history_existed"] is True
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_failure_during_res_history_projection_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    handle, captured = _inject_res_handle(monkeypatch)
    calls = []

    def exploding_project(history, target_thw, source_stream_shapes=None):
        calls.append(1)
        raise RuntimeError("history projection exploded")

    monkeypatch.setattr(
        res_multistep_adapter, "project_clean_history", exploding_project
    )
    with pytest.raises(RuntimeError, match="history projection exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(2),
            sampler_name="res_multistep", disable_pbar=True,
        )
    _assert_res_seam(captured, handle)
    # The hook only calls the projection when history exists, so the call
    # itself proves stage-0 history existed when the projection failed.
    assert calls == [1]
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


def test_failure_during_audio_transition_clears_res_state(monkeypatch):
    guider, keyframes, refs, pristine = _i2v_guider()
    handle, captured = _inject_res_handle(monkeypatch)
    observed = {}

    def exploding_reindex(*args, **kwargs):
        # Observed at the moment of explosion: history existed when the
        # audio handling failed (same first boundary as the spectral case).
        observed["history_existed"] = handle.state.old_denoised is not None
        raise RuntimeError("audio transition exploded")

    monkeypatch.setattr(h3_runtime, "clock_reindex_audio_state", exploding_reindex)
    with pytest.raises(RuntimeError, match="audio transition exploded"):
        run_speed_pipeline(
            SeededRandomNoise(), guider, SIGMAS, make_latent(),
            _explicit_ladder_cfg(2),
            sampler_name="res_multistep", disable_pbar=True,
        )
    _assert_res_seam(captured, handle)
    assert observed["history_existed"] is True
    _assert_res_state_cleared(handle)
    assert not hasattr(guider, _LW_ATTR)
    _assert_pristine_conds(keyframes, refs, pristine)


# ---------------------------------------------------------------------------
# §20 state-leak — a fresh RES run after a completed RES run starts clean
# ---------------------------------------------------------------------------

def test_res_second_generation_after_completed_run_starts_clean(monkeypatch):
    cfg = _explicit_ladder_cfg(3)
    guider = ResExecGuider()

    handle_1, out_1, _ = _run_res(cfg, guider, monkeypatch)
    _assert_res_state_cleared(handle_1)

    handle_2, out_2, _ = _run_res(cfg, guider, monkeypatch)
    # Same guider object, but a brand-new run-scoped handle: state is never
    # reused between queue executions.
    assert handle_2 is not handle_1
    # The second generation's first stage started from empty history —
    # nothing leaked from the completed first run.
    assert guider.stage_entries[3] is None
    # ... and the whole second trajectory is identical to the first
    # (deterministic engine, no carried-over corruption).
    video_1, audio_1 = out_1["samples"].unbind()
    video_2, audio_2 = out_2["samples"].unbind()
    assert torch.equal(video_1, video_2)
    assert torch.equal(audio_1, audio_2)

    # Control: a fresh guider produces the same output bit-for-bit.
    control = ResExecGuider()
    _, out_control, _ = _run_res(cfg, control, monkeypatch)
    video_c, audio_c = out_control["samples"].unbind()
    assert torch.equal(video_2, video_c)
    assert torch.equal(audio_2, audio_c)
    assert not hasattr(guider, _LW_ATTR)


# ---------------------------------------------------------------------------
# §23 zero-step torture — coincident boundaries through the real runtime
# ---------------------------------------------------------------------------

def test_res_coincident_boundary_zero_step_stages_preserve_history(monkeypatch):
    """Both transitions quantize onto schedule index 1: stage 1 runs zero
    denoising steps. The zero-step stage must not update solver history, the
    transition hooks must still project/rebase it (twice, without corruption),
    and the final stage must enter with valid history so it can resume
    second-order behavior."""
    guider = ResExecGuider()
    _, out, _ = _run_res(_automatic_calibrated_cfg(3), guider, monkeypatch)

    assert len(guider.sigma_calls) == 3
    assert len(guider.sigma_calls[1]) == 1  # zero-step middle stage
    # Model ran 1 + 0 + 9 intervals: the zero-step stage added no history.
    assert guider.model_evals == len(SIGMAS) - 1

    first = guider.model_outputs[0]
    entries = guider.stage_entries
    # Hook 1 projected stage-0 history to stage 1's geometry and rebased
    # both sigma fields (prev_sigma_in == 1.0 stays 1.0 — legal history).
    snap1 = entries[1]
    assert snap1 is not None
    assert tuple(snap1.video.shape[-3:]) == _stage_geometry(3, 1)
    assert snap1.old_sigma_down == pytest.approx(
        aligned_sigma(float(SIGMAS[1]), 2.0)[1], rel=1e-6
    )
    assert snap1.prev_sigma_in == pytest.approx(1.0)

    # Hook 2 ran on the unchanged history (zero intervals in between):
    # the result equals the independent double projection, and the clean
    # audio estimate is still the very same tensor the model produced.
    snap2 = entries[2]
    expected = project_clean_history(
        project_clean_history(first, _stage_geometry(3, 1)),
        _stage_geometry(3, 2),
    )
    expected_video = expected.unbind()[0]
    assert torch.equal(snap2.video, expected_video)
    assert snap2.audio is first.unbind()[1]
    assert snap2.audio.abs().sum() > 0
    # Sigma metadata reflects both rebases.
    chained = aligned_sigma(float(SIGMAS[1]), 2.0)[1]
    assert snap2.old_sigma_down == pytest.approx(
        aligned_sigma(chained, 1.5)[1], rel=1e-6
    )
    assert snap2.prev_sigma_in == pytest.approx(1.0)
    # Valid history at the final stage: second-order RES can resume.
    assert snap2.old_sigma_down != pytest.approx(snap2.prev_sigma_in)
    _assert_full_res_nested(out)
    assert not hasattr(guider, _LW_ATTR)


def test_res_final_stage_second_order_behavior_depends_on_carried_history(monkeypatch):
    """The §23 precondition alone (history differs from the input sigma) is
    not proof: with carried history the final stage's intervals must follow
    a different trajectory than the documented §15/§28 test-only reset
    control — history cleared at the boundary, through the real runtime.
    A run that silently reset history at the boundary would match the
    control and fail this test."""
    cfg = _explicit_ladder_cfg(2)

    class ResetAtBoundary(ResExecGuider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if len(self.sigma_calls) == 1:  # the final stage call
                sampler.state.clear()  # test-only reset-history control
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    _, out_carried, _ = _run_res(cfg, ResExecGuider(), monkeypatch)
    handle_reset, out_reset, _ = _run_res(cfg, ResetAtBoundary(), monkeypatch)
    assert handle_reset.state.old_denoised is None
    assert not torch.equal(out_carried["samples"].unbind()[0], out_reset["samples"].unbind()[0])
