"""Execution-level regression tests for global transition-step semantics.

`transition_steps` are GLOBAL indices into the original sigma schedule
(canonical SPEED). These tests run `run_speed_pipeline` with a fake guider
that records the exact sigma schedule of every `guider.sample()` call, then
assert the observed stage schedules against a canonical oracle built here.
They catch the global-vs-local index drift where a later stage applies a
global boundary to a shortened schedule.
"""

import importlib
import sys
from types import ModuleType

import pytest
import torch

from conftest import install_comfy_stubs
from speed_scripts.config import SpeedConfig
from speed_scripts.flow import aligned_sigma
from speed_scripts.h3_runtime import run_speed_pipeline

install_comfy_stubs()


def canonical_segments(sigmas, transition_steps):
    """Official SPEED scheduling semantics (pure helper, no runtime import).

    Starts are [0] + boundaries; ends are boundaries + [last sigma index].
    Stage k covers original sigma indices [starts[k], ends[k]] inclusive —
    both endpoints are sigma entries, so the stage runs ends[k] - starts[k]
    denoising steps.
    """
    starts = [0] + list(transition_steps)
    ends = list(transition_steps) + [len(sigmas) - 1]
    return [(start, end) for start, end in zip(starts, ends)]


def canonical_stage_schedules(sigmas, transition_steps, ratios):
    """Expected per-stage sigma schedules under the upstream working-sigmas model.

    Mirrors official SPEED: a working copy of the schedule is sliced by
    global boundary indices, and after every transition the boundary
    coordinate is patched in place with the kappa-aligned sigma
    (`working[end] = aligned_sigma(float(working[end]), ratio)[1]`). For
    unique boundaries this equals the aligned-entry-per-stage model; for
    coincident boundaries it makes the second transition read the
    already-aligned coordinate and align it again.
    """
    segs = canonical_segments(sigmas, transition_steps)
    working = [float(s) for s in sigmas]
    schedules = []
    for k, (start, end) in enumerate(segs):
        schedules.append([working[i] for i in range(start, end + 1)])
        if k < len(segs) - 1:
            _kappa, entry = aligned_sigma(float(working[end]), ratios[k])
            working[end] = entry
    return schedules


def stage_intervals(stage_sched, start_idx):
    """Original denoising intervals a stage covers, given its expected schedule.

    The first entry of a post-transition stage is the aligned re-entry sigma;
    it sits AT the boundary coordinate (original index start_idx), replacing
    it — it does not add a step. Later entries are consecutive originals.
    """
    idxs = [start_idx] + list(range(start_idx + 1, start_idx + len(stage_sched)))
    return list(zip(idxs[:-1], idxs[1:]))


def assert_matches_canonical(sigma_calls, sigmas, transition_steps, ratios):
    """Assert observed stage calls match the canonical global segmentation."""
    expected = canonical_stage_schedules(sigmas, transition_steps, ratios)
    assert len(sigma_calls) == len(expected), (
        f"expected {len(expected)} stages, got {len(sigma_calls)}: {sigma_calls}"
    )
    for k, (call, exp) in enumerate(zip(sigma_calls, expected)):
        assert call == pytest.approx(exp), (
            f"stage {k} sigma schedule {call} != canonical {exp}"
        )
    # Every original denoising interval exactly once, in global order.
    covered = []
    for (start, _end), call in zip(canonical_segments(sigmas, transition_steps), sigma_calls):
        covered.extend(stage_intervals(call, start))
    expected_cover = [(i, i + 1) for i in range(len(sigmas) - 1)]
    assert covered == expected_cover, (
        f"covered intervals {covered} != expected {expected_cover}"
    )


class RecordingGuider:
    """Fake guider that records every sigma schedule passed to sample()."""

    def __init__(self):
        self.sigma_calls = []
        self.model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        self.conds = {}

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.sigma_calls.append([float(s) for s in sigmas])
        if callback is not None:
            for i in range(len(sigmas) - 1):
                callback(i, latent_image, latent_image, len(sigmas) - 1)
        return latent_image


class RecordingNoise:
    """Noise that regenerates the (video, audio) streams unchanged."""

    seed = 42

    def generate_noise(self, latent):
        samples = latent.get("samples")
        if getattr(samples, "is_nested", False):
            vids = [s for s in samples.unbind() if s.ndim == 5]
            auds = [s for s in samples.unbind() if s.ndim != 5]
            return type("NT", (), {"is_nested": True, "unbind": lambda self: vids + auds})()
        return samples


def make_latent(t=2, h=8, w=8):
    video = torch.zeros(1, 1, t, h, w)
    audio = torch.zeros(1, 1, 2, 44)
    nested = type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()
    return {"samples": nested}


SIGMAS_10 = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])


def run(cfg, sigmas=SIGMAS_10, latent=None):
    guider = RecordingGuider()
    run_speed_pipeline(
        RecordingNoise(), guider, sigmas, latent if latent is not None else make_latent(), cfg,
        sampler=type("S", (), {"name": "euler"})(),
        disable_pbar=True,
    )
    return guider.sigma_calls


def test_four_stage_explicit_global_boundaries():
    """4-stage run: boundaries (3,5,8) on a 10-step schedule.

    Expected stage schedules:
      stage 0: [s0, s1, s2, s3]
      stage 1: [aligned(s3), s4, s5]
      stage 2: [aligned(s5), s6, s7, s8]
      stage 3: [aligned(s8), s9, s10]
    Stage 1 must NOT include original indices 6, 7, 8.
    """
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 0.75, 1.0),
        transition_steps=(3, 5, 8),
        transition_mode="explicit",
    )
    calls = run(cfg)
    orig = [float(s) for s in SIGMAS_10]

    assert len(calls) == 4
    # Stage 0: exact original slice [0..3].
    assert calls[0] == orig[0:4]
    # Stage 1: [aligned(s3), s4, s5] — must NOT contain s6, s7, s8.
    assert calls[1][1:] == orig[4:6]
    assert orig[6] not in calls[1] and orig[7] not in calls[1] and orig[8] not in calls[1]
    # Stage 2: [aligned(s5), s6, s7, s8].
    assert calls[2][1:] == orig[6:9]
    # Stage 3: [aligned(s8), s9, s10].
    assert calls[3][1:] == orig[9:11]

    ratios = [0.5 / 0.25, 0.75 / 0.5, 1.0 / 0.75]
    assert_matches_canonical(calls, SIGMAS_10, (3, 5, 8), ratios)


def test_three_stage_explicit_global_boundaries():
    """3-stage run: boundaries (3,7) -> stages 0->3, 3->7, 7->end."""
    cfg = SpeedConfig(
        scales=(1 / 3, 2 / 3, 1.0),
        transition_steps=(3, 7),
        transition_mode="explicit",
    )
    calls = run(cfg)
    assert len(calls) == 3
    ratios = [(2 / 3) / (1 / 3), 1.0 / (2 / 3)]
    assert_matches_canonical(calls, SIGMAS_10, (3, 7), ratios)


def test_three_stage_preset_quarter_half_full():
    """3-stage preset boundaries (3, 5) execute at original steps 3 and 5.

    Uses the same preset defaults the Automatic node ships
    (quarter_half_full -> scales (0.25, 0.5, 1.0), steps (3, 5)).
    """
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 1.0),
        transition_steps=(3, 5),
        transition_mode="explicit",
    )
    calls = run(cfg)
    assert len(calls) == 3
    orig = [float(s) for s in SIGMAS_10]
    assert calls[0] == orig[0:4]
    assert calls[1][1:] == orig[4:6]
    assert calls[2][1:] == orig[6:11]
    ratios = [0.5 / 0.25, 1.0 / 0.5]
    assert_matches_canonical(calls, SIGMAS_10, (3, 5), ratios)


def test_two_stage_behavior_unchanged():
    """2-stage run (already correct before the fix): schedule unchanged."""
    cfg = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),
        transition_mode="explicit",
    )
    calls = run(cfg)
    assert len(calls) == 2
    orig = [float(s) for s in SIGMAS_10]
    assert calls[0] == orig[0:6]
    assert calls[1][1:] == orig[6:11]
    assert_matches_canonical(calls, SIGMAS_10, (5,), [2.0])


def test_no_skipped_or_duplicated_intervals():
    """Every original denoising interval executes exactly once (4-stage)."""
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 0.75, 1.0),
        transition_steps=(3, 5, 8),
        transition_mode="explicit",
    )
    calls = run(cfg)
    ratios = [0.5 / 0.25, 0.75 / 0.5, 1.0 / 0.75]
    assert_matches_canonical(calls, SIGMAS_10, (3, 5, 8), ratios)


def test_total_callbacks_equal_schedule_steps():
    """Total denoising steps across stages == len(original_sigmas) - 1.

    The prepended aligned sigma changes the stage-entry coordinate; it does
    not create another original denoising step.
    """
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 0.75, 1.0),
        transition_steps=(3, 5, 8),
        transition_mode="explicit",
    )
    calls = run(cfg)
    total = sum(len(call) - 1 for call in calls)
    assert total == len(SIGMAS_10) - 1


def test_aligned_sigma_source_is_original_boundary():
    """Each transition's kappa alignment uses the ORIGINAL boundary sigma.

    new_q = aligned_sigma(sigmas[global_boundary], ratio)[1].
    """
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 0.75, 1.0),
        transition_steps=(3, 5, 8),
        transition_mode="explicit",
    )
    calls = run(cfg)
    orig = [float(s) for s in SIGMAS_10]

    _k0, new_q0 = aligned_sigma(orig[3], 0.5 / 0.25)
    assert calls[1][0] == pytest.approx(new_q0)
    _k1, new_q1 = aligned_sigma(orig[5], 0.75 / 0.5)
    assert calls[2][0] == pytest.approx(new_q1)
    _k2, new_q2 = aligned_sigma(orig[8], 1.0 / 0.75)
    assert calls[3][0] == pytest.approx(new_q2)


def test_delta_custom_multi_stage_matches_resolved_boundaries():
    """delta-optimal multi-stage execution matches resolve_transition_steps.

    Catches drift between planner (resolve) and executor (stage slicing).
    The runtime resolves against the LIVE latent dims, so the latent here
    matches the config's full_latent dims exactly.
    """
    from speed_scripts.h3_runtime import resolve_transition_steps

    sigmas = torch.linspace(1.0, 0.0, 21)
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 1.0),
        transition_steps=(3, 5),
        transition_mode="delta_custom",
        delta=0.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=44,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(cfg, sigmas, H_full=44, W_full=80)
    calls = run(cfg, sigmas=sigmas, latent=make_latent(t=2, h=44, w=80))

    assert len(calls) == len(tuple(resolved)) + 1
    ratios = [0.5 / 0.25, 1.0 / 0.5]
    assert_matches_canonical(calls, sigmas, tuple(resolved), ratios)


def test_delta_custom_duplicate_boundaries_follow_working_sigmas():
    """Resolved coincident boundaries execute the upstream working-sigmas model.

    With A=219.48, beta=2.42, delta=0.01 on an 8x8 latent, both thresholds
    quantize onto sigma index 1 (guarded below, so the test cannot silently
    rot if the calibration or the schedule changes). Upstream SPEED handles
    the repeated boundary as: stage 0 samples 0..1, transition A aligns and
    patches working[1], the intermediate stage gets the single-entry slice
    working[1:2] and denoises ZERO steps, transition B reads the
    already-aligned coordinate, aligns it AGAIN (intentional compounding),
    patches again, and the final stage runs working[1:] with the
    double-aligned entry.
    """
    from speed_scripts.h3_runtime import resolve_transition_steps

    sigmas = torch.linspace(1.0, 0.0, 11)
    cfg = SpeedConfig(
        scales=(0.25, 0.5, 1.0),
        transition_steps=(3, 5),  # ignored by delta_custom
        transition_mode="delta_custom",
        delta=0.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=8,
        full_latent_w=8,
    )
    resolved = resolve_transition_steps(cfg, sigmas, H_full=8, W_full=8)
    assert resolved == (1, 1), (
        f"calibration no longer resolves to duplicate boundaries: {resolved}"
    )

    calls = run(cfg, sigmas=sigmas, latent=make_latent(t=2, h=8, w=8))
    orig = [float(s) for s in sigmas]
    ratios = [0.5 / 0.25, 1.0 / 0.5]

    # Full oracle match (generalized patch-and-read model, duplicates included).
    assert_matches_canonical(calls, sigmas, resolved, ratios)

    # Stage shapes: stage 0 ends at the boundary; the intermediate stage is a
    # single-entry schedule (zero denoising steps); the final stage gets the
    # rest.
    assert len(calls) == 3
    assert calls[0] == pytest.approx(orig[0:2])
    assert len(calls[1]) == 1, f"intermediate stage must denoise zero steps: {calls[1]}"
    assert calls[2][1:] == pytest.approx(orig[2:11])

    # The final entry is the DOUBLE-aligned boundary coordinate: transition B
    # read transition A's patched sigma and aligned it again.
    _k_a, new_q_a = aligned_sigma(orig[1], ratios[0])
    assert calls[1] == pytest.approx([new_q_a])
    _k_b, new_q_b = aligned_sigma(new_q_a, ratios[1])
    assert calls[2][0] == pytest.approx(new_q_b)

    # Zero-step stage contributes nothing: total denoising callbacks are
    # still len(sigmas) - 1.
    total = sum(len(call) - 1 for call in calls)
    assert total == len(sigmas) - 1


def test_progress_advances_once_per_global_step_multistage():
    """Preview bar advances exactly 1..n_steps across a 4-stage run."""
    pbar_updates = []
    install_comfy_stubs()
    utils = sys.modules["comfy.utils"]

    class FakePbar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, value, total=None, preview=None):
            pbar_updates.append((value, total if total is not None else self.total, preview))

    utils.ProgressBar = FakePbar

    def fake_prepare_callback(model_patcher, steps, x0_output_dict=None):
        pbar = FakePbar(steps)

        def _cb(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0
            pbar.update_absolute(step + 1, total_steps, None)

        return _cb

    preview_mod = ModuleType("latent_preview")
    preview_mod.prepare_callback = fake_prepare_callback
    sys.modules["latent_preview"] = preview_mod

    h3_runtime = importlib.import_module("speed_scripts.h3_runtime")
    try:
        importlib.reload(h3_runtime)

        cfg = SpeedConfig(
            scales=(0.25, 0.5, 0.75, 1.0),
            transition_steps=(3, 5, 8),
            transition_mode="explicit",
        )
        x0_output = {}
        out, denoised = h3_runtime.run_speed_pipeline(
            RecordingNoise(), RecordingGuider(), SIGMAS_10, make_latent(), cfg,
            sampler=object(), disable_pbar=False, output_device=None,
            x0_output=x0_output,
        )
        assert out is not None and denoised is not None
        values = [v for v, _t, _p in pbar_updates]
        assert values == list(range(1, 11)), (
            f"preview bar not exactly 1..10 across 4 stages: {values}"
        )
        assert all(t == 10 for _v, t, _p in pbar_updates)
        assert "x0" in x0_output
    finally:
        sys.modules.pop("latent_preview", None)
        install_comfy_stubs()
        importlib.reload(h3_runtime)


def test_boundary_at_schedule_end_raises():
    """A boundary at the last step index is outside the interior — rejected."""
    cfg = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(10,),  # schedule has 10 steps; interior is 1..9
        transition_mode="explicit",
    )
    with pytest.raises(ValueError, match="inside the sigma schedule"):
        run(cfg)


def test_duplicate_transition_steps_rejected():
    """Explicit duplicate boundaries (5,5) are rejected as user error.

    Upstream SPEED does NOT collapse duplicate transitions: it runs a
    zero-step intermediate segment (a single-entry sigma schedule denoises
    nothing) and still performs the spectral expand + alignment for each
    transition. Resolved ("delta_custom") steps follow that model and are
    accepted. Explicit steps are the H3-facing convenience API, where a
    duplicate index cannot express a meaningful stage ladder, so it fails
    loudly at the config boundary.
    """
    with pytest.raises(ValueError, match="strictly increasing"):
        SpeedConfig(
            scales=(1 / 3, 2 / 3, 1.0),
            transition_steps=(5, 5),
            transition_mode="explicit",
        )


def test_decreasing_transition_steps_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        SpeedConfig(
            scales=(1 / 3, 2 / 3, 1.0),
            transition_steps=(7, 3),
            transition_mode="explicit",
        )


def test_duplicate_transition_steps_allowed_in_delta_custom():
    """delta_custom accepts duplicate/decreasing steps (resolved upstream-style)."""
    SpeedConfig(
        scales=(1 / 3, 2 / 3, 1.0),
        transition_steps=(5, 5),
        transition_mode="delta_custom",
    )
    SpeedConfig(
        scales=(1 / 3, 2 / 3, 1.0),
        transition_steps=(7, 3),
        transition_mode="delta_custom",
    )
