"""Sampler selection, lifecycle, transition, progress, and I2V contracts."""

import pytest
import torch

from conftest import (
    LADDER_BOUNDARIES,
    RecordingEchoGuider,
    SeededRandomNoise,
    make_fake_noise,
    make_latent,
    make_recording_guider,
)
from speed_scripts import h3_runtime
from speed_scripts.planning import STAGES_TO_SCALES
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import run_speed_pipeline
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
    SpeedTransition,
    SpeedSamplerHandle,
    _ResMultistepSamplerHandle,
    create_speed_sampler_handle,
)


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def _cfg(**kwargs):
    kwargs.setdefault("scales", (.33, .66, 1.0))
    kwargs.setdefault("transition_steps", (3, 5))
    return SpeedConfig(transition_mode="explicit", **kwargs)


def _explicit_ladder_cfg(stages):
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=LADDER_BOUNDARIES[stages],
        transition_mode="explicit",
    )


def _automatic_calibrated_cfg(stages):
    """Automatic config that produces coincident boundaries on this test schedule."""
    return SpeedConfig(
        scales=STAGES_TO_SCALES[stages],
        transition_steps=tuple(range(1, len(STAGES_TO_SCALES[stages]))),
        transition_mode="delta_custom",
        delta=.005,
        noise_amplitude=12.105,
        noise_decay_exponent=.773,
    )


def _run(sampler, cfg, guider, **kwargs):
    return run_speed_pipeline(
        SeededRandomNoise(), guider, SIGMAS, make_latent(), cfg,
        sampler_name=sampler, disable_pbar=True, **kwargs,
    )


def _assert_full_res_nested(latent):
    """Assert full-resolution nested video and audio output."""
    video, audio = latent["samples"].unbind()
    assert video.ndim == 5 and audio.ndim == 4
    assert tuple(video.shape[-2:]) == (8, 8)


def _nested(video, audio):
    class Nested:
        is_nested = True

        def __init__(self, streams):
            self.streams = list(streams)

        def unbind(self):
            return list(self.streams)

    return Nested([video, audio])


class EchoGuider:
    """Add a fixed video offset and record stage order, sampler, and input size."""

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
# Public sampler selector
# ---------------------------------------------------------------------------

def test_public_selector_is_exactly_the_five_supported_names():
    assert STATELESS_SPEED_SAMPLERS == ("euler", "heun", "dpm_2", "exp_heun_2_x0")
    assert SUPPORTED_SPEED_SAMPLERS == STATELESS_SPEED_SAMPLERS + ("res_multistep",)


@pytest.mark.parametrize("name", STATELESS_SPEED_SAMPLERS)
def test_factory_accepts_each_supported_name(name):
    handle = create_speed_sampler_handle(name)
    assert isinstance(handle, SpeedSamplerHandle)
    assert handle.capability is SamplerCapability.STATELESS_STEP_LOCAL
    # Native Comfy sampler object for that name (conftest stub shape).
    assert handle.sampler == ("sampler", name)


def test_factory_routes_res_to_stateful_handle_without_native_sampler(monkeypatch):
    import comfy.samplers

    def native_sampler_must_not_run(name):
        raise AssertionError(f"native sampler lookup was called for {name!r}")

    monkeypatch.setattr(comfy.samplers, "sampler_object", native_sampler_must_not_run)
    handle = create_speed_sampler_handle("res_multistep")
    assert isinstance(handle, _ResMultistepSamplerHandle)
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    handle.close()


@pytest.mark.parametrize("name", ["dpmpp_2m", "Euler", "euler_ancestral", ""])
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
# Euler stage behavior
# ---------------------------------------------------------------------------

def test_default_and_explicit_paths_select_euler_exactly_once_per_run(monkeypatch):
    selected = []

    def recording_factory(name, **kwargs):
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
    """Stage slices use global boundaries and aligned boundary sigmas."""
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

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: MarkerHandle())
    guider = EchoGuider()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    # Every stage uses the sampler object owned by the handle.
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
    # Ten denoising intervals produce ten global callbacks.
    assert seen == [(i, 10) for i in range(10)]


# ---------------------------------------------------------------------------
# Transition hook position — once per configured transition
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

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: HookHandle())
    guider = HookGuider(video_offset=.5)
    guider.test_events = events
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )

    scales = (.33, .66, 1.0)
    # One hook runs between each pair of stages.
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
    # The hook sees the completed source and target stage sizes.
    assert transitions_seen[0].source_thw == (2, 3, 3)
    assert transitions_seen[0].target_thw == (2, 5, 5)
    assert transitions_seen[1].source_thw == (2, 5, 5)
    assert transitions_seen[1].target_thw == (2, 8, 8)


def test_hook_failure_aborts_the_run_before_the_next_stage_samples(monkeypatch):
    """A transition-hook failure stops the next stage and still runs cleanup."""

    class ExplodingHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def on_transition(self, transition):
            if transition.stage_idx == 0:
                raise RuntimeError("hook exploded")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: ExplodingHandle())
    guider = EchoGuider()
    with pytest.raises(RuntimeError, match="hook exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    assert guider.events.count("sample") == 1


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

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: HookHandle())
    # Both transitions resolve to the same schedule index.
    cfg = SpeedConfig(
        scales=(.25, .5, 1.0),
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
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
# Sampler and I2V lifecycle
# ---------------------------------------------------------------------------

def test_cleanup_closes_sampler_and_restores_on_success(monkeypatch):
    order = []

    class CleanupHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            order.append("close")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: CleanupHandle())
    monkeypatch.setattr(
        h3_runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    guider = make_recording_guider()
    run_speed_pipeline(
        make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
        disable_pbar=True,
    )
    # Restore before the final stage, then restore again during cleanup.
    assert order == ["apply_final", "close", "apply_final"]


def test_cleanup_still_restores_when_sampler_close_fails(monkeypatch):
    order = []

    class FailingHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            order.append("close")
            raise RuntimeError("close exploded")

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: FailingHandle())
    monkeypatch.setattr(
        h3_runtime.LatentWalker, "apply_final", lambda self: order.append("apply_final")
    )
    guider = make_recording_guider()
    with pytest.raises(RuntimeError, match="close exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    # Keyframes are restored even when sampler cleanup fails.
    assert order == ["apply_final", "close", "apply_final"]


def test_cleanup_still_closes_sampler_when_restore_fails(monkeypatch):
    """Sampler cleanup runs even when keyframe restore fails."""
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

    monkeypatch.setattr(h3_runtime, "create_speed_sampler_handle", lambda name, **kwargs: CleanupHandle())
    monkeypatch.setattr(h3_runtime.LatentWalker, "apply_final", exploding_final)
    guider = make_recording_guider()
    # Cleanup closes the sampler and retries keyframe restore.
    with pytest.raises(RuntimeError, match="restore exploded"):
        run_speed_pipeline(
            make_fake_noise(), guider, SIGMAS, make_latent(), _cfg(),
            disable_pbar=True,
        )
    assert order == ["apply_final", "close", "apply_final"]


# ---------------------------------------------------------------------------
# Runtime validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Noise policies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_noise_policy_direct_coarse_smoke_per_sampler(sampler):
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(sampler, _explicit_ladder_cfg(3), guider)
    assert len(guider.noise_shapes) == 3
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)



def test_coupled_full_grid_transforms_full_noise_once(monkeypatch):
    original = h3_runtime.dct_temporal
    full_grid_calls = []

    def recording_dct_temporal(value):
        if tuple(value.shape[-2:]) == (8, 8):
            full_grid_calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(h3_runtime, "dct_temporal", recording_dct_temporal)
    cfg = _cfg(noise_policy="coupled_full_grid")
    _run("euler", cfg, RecordingEchoGuider(video_offset=.5))
    assert len(full_grid_calls) == 1

@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_noise_policy_coupled_full_grid_smoke_per_sampler(sampler):
    cfg = _cfg(noise_policy="coupled_full_grid")
    guider = RecordingEchoGuider(video_offset=.5)
    out, denoised = _run(sampler, cfg, guider)
    assert len(guider.noise_shapes) == 3
    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)


# ---------------------------------------------------------------------------
# Progress and preview
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_public_progress_is_monotonic_across_stages(sampler):
    """Callback indices use one continuous global timeline."""
    seen = []
    guider = RecordingEchoGuider(video_offset=.5)
    _run(
        sampler, _explicit_ladder_cfg(3), guider,
        preview_callback=lambda step, x0, x, total: seen.append(step),
    )
    assert seen == list(range(10))



@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_shared_x0_output_and_final_denoised_are_valid_nested_h3(sampler):
    """The shared x0 dict stays valid across stages and the denoised output
    keeps the full nested H3 video+audio structure at full resolution."""
    x0_output = {}
    guider = RecordingEchoGuider(video_offset=.5)
    _out, denoised = _run(
        sampler, _explicit_ladder_cfg(3), guider, x0_output=x0_output,
    )
    assert "x0" in x0_output
    _assert_full_res_nested(denoised)
    _assert_full_res_nested({"samples": x0_output["x0"]})


# ---------------------------------------------------------------------------
# Coincident boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_turbo_coincident_ladder_runs_every_transition_and_alignment(sampler):
    """Coincident boundaries still run every transition and sigma alignment."""
    cfg = _automatic_calibrated_cfg(3)
    guider = RecordingEchoGuider()
    seen = []
    out, _ = _run(
        sampler, cfg, guider,
        preview_callback=lambda step, x0, x, total: seen.append(step),
    )

    assert len(guider.sigma_calls) == 3
    for call in guider.sigma_calls[1:-1]:
        assert len(call) == 1  # single-sigma slice: no crash, zero steps
    # All three stages use the selected sampler.
    assert guider.samplers == [("sampler", sampler)] * 3
    # Each transition aligns the shared boundary for its own stage ratio.
    first_kappa, first_aligned = aligned_sigma(float(SIGMAS[1]), 2.0)
    second_kappa, second_aligned = aligned_sigma(first_aligned, 1.5)
    assert guider.sigma_calls[1][0] == pytest.approx(first_aligned)
    assert guider.sigma_calls[2][0] == pytest.approx(second_aligned)
    # Progress remains monotonic.
    assert seen == list(range(10))
    _assert_full_res_nested(out)


# ---------------------------------------------------------------------------
# I2V keyframe lifecycle
# ---------------------------------------------------------------------------

def _i2v_guider():
    """RecordingEchoGuider with real I2V-shaped conditioning attached.

    Returns (guider, keyframes, refs, original_conds) where keyframes/refs
    are the live holder dicts the walker resizes and original_conds is the
    container object the guider owns.
    """
    guider = RecordingEchoGuider(video_offset=.5)
    keyframes = [{"latent": torch.zeros(1, 1, 2, 8, 8)} for _ in range(2)]
    refs = [{"latent": torch.zeros(1, 1, 2, 8, 8)} for _ in range(2)]
    guider.original_conds = {
        "positive": [{
            "minimax_keyframes": keyframes,
            "minimax_refs": refs,
        }],
        "negative": [],
    }
    return guider, keyframes, refs, guider.original_conds


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_i2v_smoke_and_pristine_restore_on_success(sampler):
    guider, keyframes, refs, original_conds = _i2v_guider()
    out, denoised = _run(sampler, _explicit_ladder_cfg(3), guider)

    _assert_full_res_nested(out)
    _assert_full_res_nested(denoised)
    # Keyframes return to their original full resolution.
    for kf in keyframes:
        assert tuple(kf["latent"].shape[-2:]) == (8, 8)
    # Reference latents stay at full resolution.
    for ref in refs:
        assert tuple(ref["latent"].shape[-2:]) == (8, 8)
    # Restore the same conditioning containers and holder objects.
    assert guider.original_conds is original_conds
    cond = original_conds["positive"][0]
    assert cond["minimax_keyframes"] is keyframes
    assert cond["minimax_refs"] is refs


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_i2v_failure_restores_pristine_conditioning(sampler):
    guider, keyframes, refs, original_conds = _i2v_guider()

    import speed_scripts.h3_runtime as rt

    original_expand = rt.spectral_expand

    def exploding_expand(value, target_hw, sigma, seed):
        raise RuntimeError("i2v forced failure")

    rt.spectral_expand = exploding_expand
    try:
        with pytest.raises(RuntimeError, match="i2v forced failure"):
            _run(sampler, _explicit_ladder_cfg(3), guider)
    finally:
        rt.spectral_expand = original_expand

    # Failure after downscaling still restores the original keyframes.
    for kf in keyframes:
        assert tuple(kf["latent"].shape[-2:]) == (8, 8)
    for ref in refs:
        assert tuple(ref["latent"].shape[-2:]) == (8, 8)
    assert guider.original_conds is original_conds


# ---------------------------------------------------------------------------
# Failure cleanup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_stage_failure_closes_handle(sampler):
    """A second-stage failure still closes the sampler handle."""
    closed = []
    events = []

    class FailingCloseHandle(SpeedSamplerHandle):
        def __init__(self):
            self.sampler = object()
            self.capability = SamplerCapability.STATELESS_STEP_LOCAL

        def close(self):
            closed.append(True)

    # Raise from the second stage call.
    class ExplodingStageGuider(RecordingEchoGuider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            events.append("sample")
            if len(events) == 2:  # second stage
                raise RuntimeError("stage exploded")
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    guider = ExplodingStageGuider()
    _original_factory = h3_runtime.create_speed_sampler_handle
    h3_runtime.create_speed_sampler_handle = lambda name, **kwargs: FailingCloseHandle()
    try:
        with pytest.raises(RuntimeError, match="stage exploded"):
            _run(sampler, _explicit_ladder_cfg(3), guider)
    finally:
        h3_runtime.create_speed_sampler_handle = _original_factory

    assert closed == [True]
    assert events == ["sample", "sample"]  # stage 2 never ran



@pytest.mark.parametrize("sampler", STATELESS_SPEED_SAMPLERS)
def test_second_generation_after_failure_starts_clean(sampler):
    """The same guider starts the next generation with clean state."""
    class ExplodingStageGuider(RecordingEchoGuider):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.remaining_stage_failures = 0

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if self.remaining_stage_failures > 0:
                self.remaining_stage_failures -= 1
                raise RuntimeError("first generation exploded")
            return super().sample(
                noise, latent_image, sampler, sigmas, callback=callback, **kwargs
            )

    cfg = _explicit_ladder_cfg(3)

    # Generation 1: fail in the first stage call.
    guider_a = ExplodingStageGuider()
    guider_a.remaining_stage_failures = 1
    with pytest.raises(RuntimeError, match="first generation exploded"):
        _run(sampler, cfg, guider_a)

    # The next generation matches a fresh control run.
    out_second, _ = _run(sampler, cfg, guider_a)

    guider_control = ExplodingStageGuider()
    out_control, _ = _run(sampler, cfg, guider_control)

    video_second, audio_second = out_second["samples"].unbind()
    video_control, audio_control = out_control["samples"].unbind()
    assert torch.equal(video_second, video_control)
    assert torch.equal(audio_second, audio_control)
    _assert_full_res_nested(out_second)
