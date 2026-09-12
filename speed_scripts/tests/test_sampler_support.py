"""Sampler-support contracts: public selector, run-scoped handle lifecycle,
and Euler regression through the new handle layer.

The selector tests pin the fail-closed public surface. The Euler regression
tests replay the same fake-model run the pre-handle suite used and assert the
sigma slices, transition boundaries, aligned-sigma patching, stage geometry,
deterministic output, and callback count are unchanged. The lifecycle tests
pin the hook position (once per configured transition, never per denoising
step, never after the final stage) and the nested run-level cleanup order.

Instrumented handles for the hook and cleanup tests enter through a patched
``create_speed_sampler_handle`` in the runtime module namespace, so they
exercise the same run-scoped handle path production uses; the override seam
is covered separately.
"""

import pytest
import torch

from conftest import make_fake_noise, make_latent, make_recording_guider
from speed_scripts import h3_runtime
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import _LW_ATTR, run_speed_pipeline
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
    SpeedTransition,
    SpeedSamplerHandle,
    create_speed_sampler_handle,
)


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def _cfg(**kwargs):
    kwargs.setdefault("scales", (.33, .66, 1.0))
    kwargs.setdefault("transition_steps", (3, 5))
    return SpeedConfig(transition_mode="explicit", **kwargs)


def _nested(video, audio):
    return type(
        "Nested",
        (),
        {"is_nested": True, "unbind": lambda self: [video, audio]},
    )()


class EchoGuider:
    """Additive-echo guider (public = noise video + offset) with an event log.

    Mirrors the pre-handle additive test guider: the public output is the
    stage's public noise plus a fixed video offset. Records an ordered event
    log, every sampler object it receives, and the video geometry of each
    noise argument (the coarse/re-entry noise the stage consumes), so tests
    can order the runtime's hook calls against the stage sampling calls.
    """

    class Model:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

        def process_latent_out(self, x):
            return x

    def __init__(self, video_offset=0.0):
        self.model_patcher = type("P", (), {"model": self.Model()})()
        self.video_offset = video_offset
        self.events = []
        self.samplers = []
        self.noise_shapes = []

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.events.append("sample")
        self.samplers.append(sampler)
        pub_video, pub_audio = list(noise.unbind())
        self.noise_shapes.append(tuple(pub_video.shape))
        out_video = pub_video + self.video_offset
        out = _nested(out_video, pub_audio)
        count = len(sigmas) - 1
        if callback is not None:
            for i in range(count):
                callback(i, out, out, count)
        return out


class HookGuider(EchoGuider):
    """EchoGuider that logs its sample calls into a test-owned event list."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_events = None

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        if self.test_events is not None:
            self.test_events.append("sample")
        return super().sample(
            noise, latent_image, sampler, sigmas, callback=callback, **kwargs
        )


# ---------------------------------------------------------------------------
# Public selector (source §9 "Public selector")
# ---------------------------------------------------------------------------

def test_public_selector_is_exactly_the_four_stateless_names():
    assert STATELESS_SPEED_SAMPLERS == ("euler", "heun", "dpm_2", "exp_heun_2_x0")
    assert SUPPORTED_SPEED_SAMPLERS == STATELESS_SPEED_SAMPLERS


@pytest.mark.parametrize("name", STATELESS_SPEED_SAMPLERS)
def test_factory_accepts_each_supported_name(name):
    handle = create_speed_sampler_handle(name)
    assert isinstance(handle, SpeedSamplerHandle)
    assert handle.capability is SamplerCapability.STATELESS_STEP_LOCAL
    # Native Comfy sampler object for that name (conftest stub shape).
    assert handle.sampler == ("sampler", name)


@pytest.mark.parametrize("name", ["res_multistep", "dpmpp_2m", "Euler", "euler_ancestral", ""])
def test_factory_rejects_unknown_names_fail_closed(name):
    with pytest.raises(ValueError) as excinfo:
        create_speed_sampler_handle(name)
    message = str(excinfo.value)
    assert name in message
    for supported in SUPPORTED_SPEED_SAMPLERS:
        assert supported in message


def test_base_handle_is_an_inert_noop():
    handle = SpeedSamplerHandle()
    assert handle.on_transition(
        SpeedTransition(
            stage_idx=0, ratio=2.0, old_sigma=.7, new_sigma=.5,
            source_thw=(2, 3, 3), target_thw=(2, 5, 5),
        )
    ) is None
    assert handle.close() is None


# ---------------------------------------------------------------------------
# Euler regression (source §9 "Euler regression")
# ---------------------------------------------------------------------------

def test_default_and_explicit_paths_select_euler_exactly_once_per_run(monkeypatch):
    selected = []

    def recording_factory(name):
        selected.append(name)
        return create_speed_sampler_handle(name)

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", recording_factory)
    for kwargs in ({}, {"sampler_name": "euler"}):
        selected.clear()
        guider = make_recording_guider()
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True, **kwargs,
        )
        assert selected == ["euler"]


def test_euler_sigma_slices_and_boundaries_are_unchanged():
    """Stage slices use GLOBAL boundaries and the boundary coordinate is
    patched in place with the kappa-aligned sigma."""
    calls = []
    guider = make_recording_guider(sigma_calls=calls)
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )

    working = [float(s) for s in SIGMAS]
    boundaries = (3, 5)
    scales = (.33, .66, 1.0)
    expected = []
    for stage, end in enumerate(boundaries):
        start = boundaries[stage - 1] if stage else 0
        expected.append(working[start:end + 1])
        ratio = scales[stage + 1] / scales[stage]
        _, working[end] = aligned_sigma(working[end], ratio)
    expected.append(working[boundaries[-1]:])

    assert len(calls) == 3
    for actual, wanted in zip(calls, expected):
        assert actual == pytest.approx(wanted)
    assert sum(len(call) - 1 for call in calls) == len(SIGMAS) - 1


def test_euler_routes_the_handle_sampler_into_every_guider_call(monkeypatch):
    marker = object()

    class MarkerHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = marker
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: MarkerHandle())
    guider = EchoGuider()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    # The coarse stages and the final stage all sampled through the handle's
    # sampler object — the old hardcoded euler construction is gone.
    assert len(guider.samplers) == 3
    assert all(sampler is marker for sampler in guider.samplers)


def test_euler_stage_geometry_is_unchanged():
    """Stage cond geometry follows the configured scale ladder and each
    stage consumes noise at its own grid: coarse 3x3, expanded 5x5, final 8x8."""
    stage_shapes = []
    guider = make_recording_guider(stage_shapes=stage_shapes)
    noise_guider = EchoGuider()
    latent = make_latent()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, latent, _cfg(),
        disable_pbar=True,
    )
    run_speed_pipeline(
        make_fake_noise(), noise_guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    # round(8 * .33) = 3, round(8 * .66) = 5, final stage at full 8; the
    # noise video keeps its leading batch/channel dims (1, 1, t, h, w).
    assert stage_shapes == [(3, 3), (5, 5), (8, 8)]
    assert noise_guider.noise_shapes == [(1, 1, 2, 3, 3), (1, 1, 2, 5, 5), (1, 1, 2, 8, 8)]


def test_euler_run_is_deterministic():
    """Same seed and same fake model: two identical runs produce the same
    final video latent."""
    latent = make_latent()
    out, _ = run_speed_pipeline(
        make_fake_noise(), EchoGuider(video_offset=.5), SIGMAS, latent, _cfg(),
        disable_pbar=True,
    )
    in_video, in_audio = latent["samples"].unbind()
    out_video, out_audio = out["samples"].unbind()
    assert out_video.shape == in_video.shape
    assert out_audio.shape == in_audio.shape
    out_again, _ = run_speed_pipeline(
        make_fake_noise(), EchoGuider(video_offset=.5), SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    again_video, _ = out_again["samples"].unbind()
    assert torch.equal(out_video, again_video)


def test_euler_callback_count_equals_global_denoising_intervals():
    seen = []
    guider = EchoGuider(video_offset=.5)
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
        preview_callback=lambda step, x0, x, total: seen.append((step, total)),
    )
    # A (3, 5) ladder over a 10-interval schedule forwards exactly 10
    # callbacks on one continuous global timeline — the pre-handle count.
    assert seen == [(i, 10) for i in range(10)]


# ---------------------------------------------------------------------------
# Hook position (source §7) — once per configured transition
# ---------------------------------------------------------------------------

def test_transition_hook_fires_once_per_transition_at_the_documented_position(monkeypatch):
    events = []
    transitions_seen = []

    class HookHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def on_transition(self, transition):
            events.append("hook")
            transitions_seen.append(transition)

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: HookHandle())
    guider = HookGuider(video_offset=.5)
    guider.test_events = events
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )

    scales = (.33, .66, 1.0)
    # Exactly the two configured transitions: not per denoising step (10
    # intervals) and not after the final stage (no third hook). Each hook
    # sits between its stage's guider.sample and the next stage's sample.
    assert events == ["sample", "hook", "sample", "hook", "sample"]
    assert [t.stage_idx for t in transitions_seen] == [0, 1]
    assert [t.ratio for t in transitions_seen] == [
        pytest.approx(scales[1] / scales[0]), pytest.approx(scales[2] / scales[1]),
    ]
    # Each hook reads the boundary coordinate it is aligning: stage 0 reads
    # schedule index 3 (0.7), stage 1 reads index 5 (0.5); new_sigma is the
    # aligned coordinate patched into the working schedule.
    _, new0 = aligned_sigma(.7, scales[1] / scales[0])
    _, new1 = aligned_sigma(.5, scales[2] / scales[1])
    assert transitions_seen[0].old_sigma == pytest.approx(.7)
    assert transitions_seen[0].new_sigma == pytest.approx(new0)
    assert transitions_seen[1].old_sigma == pytest.approx(.5)
    assert transitions_seen[1].new_sigma == pytest.approx(new1)
    # Spectral transition ran before the hook: source geometry is this
    # stage's grid, target geometry is the next stage's grid.
    assert transitions_seen[0].source_thw == (2, 3, 3)
    assert transitions_seen[0].target_thw == (2, 5, 5)
    assert transitions_seen[1].source_thw == (2, 5, 5)
    assert transitions_seen[1].target_thw == (2, 8, 8)


def test_hook_failure_aborts_the_run_before_the_next_stage_samples(monkeypatch):
    """The hook belongs to the gap between two stages: when it raises, the
    next stage never samples — and run-level cleanup still runs."""

    class ExplodingHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def on_transition(self, transition):
            if transition.stage_idx == 0:
                raise RuntimeError("hook exploded")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: ExplodingHandle())
    guider = EchoGuider()
    with pytest.raises(RuntimeError, match="hook exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    assert guider.events.count("sample") == 1
    assert not hasattr(guider, _LW_ATTR)


def test_coincident_boundaries_still_call_the_hook_once_each(monkeypatch):
    transitions_seen = []
    events = []

    class HookHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def on_transition(self, transition):
            events.append("hook")
            transitions_seen.append(transition.stage_idx)

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: HookHandle())
    # delta_custom quantizes both transitions onto the same schedule index
    # (the existing suite pins boundaries == (1, 1) for these parameters) —
    # the legal coincident-boundary case.
    cfg = SpeedConfig(
        scales=(.25, .5, 1.0),
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=8,
        full_latent_w=8,
    )
    guider = HookGuider()
    guider.test_events = events
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), cfg,
        disable_pbar=True,
    )
    # Both transitions fire at the coincident boundary, each between its own
    # stage's sample calls; the zero-step intermediate stage still runs.
    assert transitions_seen == [0, 1]
    assert events == ["sample", "hook", "sample", "hook", "sample"]


# ---------------------------------------------------------------------------
# Run-scoped lifecycle (source §6 cleanup structure)
# ---------------------------------------------------------------------------

def test_cleanup_runs_close_then_restore_then_drop_on_success(monkeypatch):
    order = []

    class CleanupHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            order.append("close")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: CleanupHandle())
    monkeypatch.setattr(
        h3_runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    guider = make_recording_guider()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    # apply_final also runs once before the final stage (pre-final restore);
    # the run-level cleanup then closes the sampler handle, restores again,
    # and drops the walker from the guider.
    assert order == ["apply_final", "close", "apply_final"]
    assert not hasattr(guider, _LW_ATTR)


def test_cleanup_still_restores_and_drops_when_sampler_close_fails(monkeypatch):
    order = []

    class FailingHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            order.append("close")
            raise RuntimeError("close exploded")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: FailingHandle())
    monkeypatch.setattr(
        h3_runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    guider = make_recording_guider()
    with pytest.raises(RuntimeError, match="close exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    # The failing close did not stop the walker restore (which also runs
    # once earlier, before the final stage), and the walker was still
    # dropped from the guider.
    assert order == ["apply_final", "close", "apply_final"]
    assert not hasattr(guider, _LW_ATTR)


def test_cleanup_still_drops_the_walker_when_the_restore_fails(monkeypatch):
    """If walker.apply_final() itself raises, _drop_walker() must still run —
    the innermost finally of the required cleanup structure."""
    order = []

    class CleanupHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            order.append("close")

    def exploding_final(self):
        order.append("apply_final")
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name: CleanupHandle())
    monkeypatch.setattr(h3_runtime.LatentWalker, "apply_final", exploding_final)
    guider = make_recording_guider()
    # The restore also runs once before the final stage; the FIRST failing
    # restore aborts the run there. The close still ran after it, and the
    # walker was still dropped from the guider.
    with pytest.raises(RuntimeError, match="restore exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    assert order == ["apply_final", "close", "apply_final"]
    assert not hasattr(guider, _LW_ATTR)


# ---------------------------------------------------------------------------
# Override seam and fail-closed runtime validation
# ---------------------------------------------------------------------------

def test_override_seam_wraps_the_injected_sampler_as_a_noop_handle():
    injected = object()
    guider = EchoGuider()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        sampler_override=injected, disable_pbar=True,
    )
    assert guider.samplers[0] is injected
    assert guider.samplers[1] is injected
    assert guider.samplers[2] is injected


def test_override_seam_rejects_combination_with_explicit_sampler_name():
    with pytest.raises(ValueError):
        run_speed_pipeline(
            make_fake_noise(), make_recording_guider(), SIGMAS, make_latent(),
            _cfg(), sampler_name="heun", sampler_override=object(),
            disable_pbar=True,
        )


def test_run_rejects_unsupported_sampler_name_fail_closed():
    with pytest.raises(ValueError) as excinfo:
        run_speed_pipeline(
            make_fake_noise(), make_recording_guider(), SIGMAS, make_latent(),
            _cfg(), sampler_name="dpmpp_2m", disable_pbar=True,
        )
    message = str(excinfo.value)
    assert "dpmpp_2m" in message
    for supported in SUPPORTED_SPEED_SAMPLERS:
        assert supported in message
