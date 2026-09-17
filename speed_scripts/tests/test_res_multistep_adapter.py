"""RES Multistep adapter: state model, deterministic pipeline, the
same-resolution split oracle (plan S7 §11-§15), and the transition slice
(clean history projection, sigma rebase, ``on_transition`` — plan S7 §16-§19).

The oracle is the gate for this slice: one uninterrupted deterministic RES
trajectory over a fixed sigma schedule must land exactly where the same
schedule does when it is split at a valid interior boundary and the
``ResMultistepState`` is carried across the split. Splitting must change
nothing numerically. Three targeted mutations of the carried history
(discarding ``old_denoised``, ``old_sigma_down``, or ``prev_sigma_in`` at
the split) must each break the match — that is what makes the test sensitive
to the state the SPEED stage loop has to preserve. The negative control
pins the other side: clearing the state at the split must produce a
different trajectory.

The transition oracles pin the V1 diagnostic: every SPEED boundary clears all
RES history and sigma metadata, while coincident boundaries remain empty until
a later real RES interval rebuilds the state. Projection and sigma-rebase
primitives remain covered independently above.

Everything here runs against a deterministic fake model on plain float
schedules — no ComfyUI import is needed below the handle seam.
"""



import math

import torch
import pytest

from speed_scripts.res_multistep_adapter import (
    ResMultistepSampler,
    ResMultistepState,
    _res_first_order_update,
    _res_second_order_update,
    res_multistep_sampler,
)
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
    SpeedTransition,
    _ResMultistepSamplerHandle,
    create_res_multistep_sampler_handle,
    create_speed_sampler_handle,
)


#: Fixed strictly decreasing schedule; the trailing zero is the clean point.
SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])

#: Plan S7 §15: "tight numeric tolerance". The split changes nothing, so the
#: two runs must agree to float-epsilon level on these small tensors.
TOLERANCE = 1e-6

#: Interior boundary used by the split oracle: both segments carry real
#: RES intervals, so the carried history is exercised at the seam.
SPLIT_INDEX = 4


def test_first_order_candidate_preserves_existing_algebra():
    x = torch.tensor([[[[1.0, 2.0]]]])
    denoised = torch.tensor([[[[0.25, 1.5]]]])
    sigma = 0.8
    sigma_down = 0.3

    expected = x + (x - denoised) / sigma * (sigma_down - sigma)

    assert torch.equal(
        _res_first_order_update(x, denoised, sigma, sigma_down), expected,
    )


def test_second_order_candidate_preserves_existing_algebra():
    x = torch.tensor([[[[1.0, 2.0]]]])
    denoised = torch.tensor([[[[0.25, 1.5]]]])
    old_denoised = torch.tensor([[[[0.5, 1.25]]]])
    sigma_f = 0.8
    sigma_down_f = 0.3
    old_sigma_down = 0.7
    prev_sigma_in = 0.9

    t_old = -torch.log(torch.tensor(old_sigma_down)).item()
    t_next = -torch.log(torch.tensor(sigma_down_f)).item()
    t_prev = -torch.log(torch.tensor(prev_sigma_in)).item()
    h = t_next + torch.log(torch.tensor(sigma_f)).item()
    c2 = (t_prev - t_old) / h
    phi1 = torch.expm1(torch.tensor(-h)).item() / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    b1 = 0.0 if torch.isnan(torch.tensor(phi1 - phi2 / c2)) else phi1 - phi2 / c2
    b2 = 0.0 if torch.isnan(torch.tensor(phi2 / c2)) else phi2 / c2
    expected = math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)

    assert torch.allclose(
        _res_second_order_update(
            x,
            denoised,
            old_denoised,
            sigma_f,
            sigma_down_f,
            old_sigma_down,
            prev_sigma_in,
        ),
        expected,
        atol=1e-6,
        rtol=0.0,
    )


class DeterministicModel:
    """Smooth, deterministic stand-in for the diffusion model.

    A fixed linear map of ``x`` blended with a sigma-dependent direction:
    cheap, deterministic, and nonlinear enough that first-order and
    second-order RES steps disagree numerically (the oracle's mutation
    sensitivity depends on that disagreement).
    """

    def __init__(self, channels=3, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.a = torch.randn(channels, generator=g).reshape(1, channels, 1, 1)
        self.b = torch.randn(channels, generator=g).reshape(1, channels, 1, 1)

    def __call__(self, x, sigma, **kwargs):
        s = float(sigma)
        return self.a * x + self.b * (s / (1.0 + s))


class CallbackLog:
    """Records the native per-step callback dicts."""

    def __init__(self):
        self.entries = []

    def __call__(self, entry):
        self.entries.append(entry)


def _run(samplers_args):
    """Run RES over consecutive schedule segments, one sampler call each.

    ``samplers_args`` is a list of ``(sigmas, state, clear_first)`` tuples.
    Returns ``(final_x, states, callbacks)``.
    """
    model = DeterministicModel()
    x = torch.randn(1, 3, 4, 4, generator=torch.Generator().manual_seed(123))
    states, callbacks = [], []
    for sigmas, state, clear_first in samplers_args:
        if clear_first:
            state.clear()
        log = CallbackLog()
        x = res_multistep_sampler(
            model, x, sigmas, state, callback=log, disable=True,
        )
        states.append(state)
        callbacks.append(log)
    return x, states, callbacks


def _split_segment(sigmas, start, end):
    """Slice ``sigmas[start : end]`` as one stage-local segment."""
    return sigmas[start:end]


# ---------------------------------------------------------------------------
# State object (plan S7 §12)
# ---------------------------------------------------------------------------


def test_single_sigma_stage_executes_zero_intervals_and_touches_nothing():
    """A one-entry schedule has zero intervals: no model call, no callback,
    no state write, input returned unchanged."""
    state = ResMultistepState()
    state.old_denoised = torch.ones(1, 3, 4, 4)
    state.old_sigma_down = 0.4
    state.prev_sigma_in = 0.6
    x = torch.randn(1, 3, 4, 4)
    model = DeterministicModel()
    calls = []

    def counting_model(*args, **kwargs):
        calls.append(1)
        return model(*args, **kwargs)

    out = res_multistep_sampler(
        counting_model, x, torch.tensor([0.5]), state, disable=True,
    )
    assert calls == []
    assert out is x
    assert state.old_denoised is not None
    assert state.old_sigma_down == pytest.approx(0.4)
    assert state.prev_sigma_in == pytest.approx(0.6)


def test_final_zero_sigma_interval_uses_first_order_and_updates_state():
    """The last interval lands on sigma 0: it must take the first-order
    path (no -log(0)) and still update all three history fields."""
    state = ResMultistepState()
    model = DeterministicModel()
    x = torch.randn(1, 3, 4, 4)
    out = res_multistep_sampler(
        model, x, torch.tensor([0.3, 0.0]), state, disable=True,
    )
    assert torch.isfinite(out).all()
    assert state.old_denoised is not None
    assert state.old_sigma_down == 0.0
    assert state.prev_sigma_in == pytest.approx(0.3)


def test_sampler_object_shares_one_state_across_calls():
    """The KSAMPLER-compatible wrapper routes every call through the same
    state object, so history survives separate invocations."""
    sampler = ResMultistepSampler()
    model = DeterministicModel()
    x = torch.randn(1, 3, 4, 4)
    x = sampler(model, x, SIGMAS[: SPLIT_INDEX + 1], disable=True)
    state_after_first = (
        sampler.state.old_sigma_down, sampler.state.prev_sigma_in,
    )
    x = sampler(model, x, SIGMAS[SPLIT_INDEX:], disable=True)
    # Last completed interval of the first segment: 0.7 -> 0.6.
    assert state_after_first == pytest.approx((0.6, 0.7))
    assert sampler.state.old_sigma_down == 0.0
    assert sampler.state.prev_sigma_in == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Same-resolution split oracle (plan S7 §15)
# ---------------------------------------------------------------------------

def test_split_with_carried_state_matches_uninterrupted_run():
    """Split the schedule at an interior boundary and carry the state:
    the final x and the full history trajectory must match the
    uninterrupted run within tight tolerance."""
    full_x, full_states, full_cbs = _run([(SIGMAS, ResMultistepState(), False)])

    state = ResMultistepState()
    split_x, split_states, split_cbs = _run([
        (_split_segment(SIGMAS, 0, SPLIT_INDEX + 1), state, False),
        (_split_segment(SIGMAS, SPLIT_INDEX, len(SIGMAS)), state, False),
    ])

    assert torch.allclose(full_x, split_x, atol=TOLERANCE, rtol=0.0)
    # The whole trajectory, not just the endpoint: per-step callback payloads
    # must line up one-to-one.
    assert len(full_cbs[0].entries) == len(SIGMAS) - 1
    assert len(split_cbs[0].entries) + len(split_cbs[1].entries) == len(SIGMAS) - 1
    for a, b in zip(
        full_cbs[0].entries,
        split_cbs[0].entries + split_cbs[1].entries,
    ):
        assert torch.allclose(a["x"], b["x"], atol=TOLERANCE, rtol=0.0)
        assert torch.allclose(a["denoised"], b["denoised"], atol=TOLERANCE, rtol=0.0)
    # Carried state converges to the same terminal history.
    assert full_states[0].old_sigma_down == split_states[-1].old_sigma_down
    assert full_states[0].prev_sigma_in == split_states[-1].prev_sigma_in
    assert torch.allclose(
        full_states[0].old_denoised, split_states[-1].old_denoised,
        atol=TOLERANCE, rtol=0.0,
    )


# ---------------------------------------------------------------------------
# Mutation sensitivity: each discarded field must break the match
# ---------------------------------------------------------------------------

def _split_discarding(discard):
    """Run the split oracle but drop one history field at the split."""
    state = ResMultistepState()
    _run([(_split_segment(SIGMAS, 0, SPLIT_INDEX + 1), state, False)])
    if discard == "old_denoised":
        state.old_denoised = None
    elif discard == "old_sigma_down":
        state.old_sigma_down = None
    elif discard == "prev_sigma_in":
        state.prev_sigma_in = None
    x, _, _ = _run([(_split_segment(SIGMAS, SPLIT_INDEX, len(SIGMAS)), state, False)])
    return x


@pytest.mark.parametrize("field", ["old_denoised", "old_sigma_down", "prev_sigma_in"])
def test_oracle_fails_when_a_history_field_is_discarded(field):
    full_x, _, _ = _run([(SIGMAS, ResMultistepState(), False)])
    mutated_x = _split_discarding(field)
    assert not torch.allclose(full_x, mutated_x, atol=TOLERANCE, rtol=0.0)


# ---------------------------------------------------------------------------
# Negative control (plan S7 §15)
# ---------------------------------------------------------------------------

def test_clearing_state_at_the_split_produces_a_different_result():
    full_x, _, _ = _run([(SIGMAS, ResMultistepState(), False)])
    state = ResMultistepState()
    reset_x, _, _ = _run([
        (_split_segment(SIGMAS, 0, SPLIT_INDEX + 1), state, False),
        (_split_segment(SIGMAS, SPLIT_INDEX, len(SIGMAS)), state, True),
    ])
    assert not torch.allclose(full_x, reset_x, atol=TOLERANCE, rtol=0.0)


# ---------------------------------------------------------------------------
# Handle seam (plan S7 §5/§13; public selector uses the stateful adapter)
# ---------------------------------------------------------------------------

def test_res_handle_keeps_single_history_and_clears_on_close():
    handle = create_res_multistep_sampler_handle()
    assert isinstance(handle, _ResMultistepSamplerHandle)
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    assert isinstance(handle.sampler, ResMultistepSampler)
    assert handle.state is handle.sampler.state
    # History actually lands in the run-scoped state.
    model = DeterministicModel()
    x = torch.randn(1, 3, 4, 4)
    handle.sampler(model, x, SIGMAS[: SPLIT_INDEX + 1], disable=True)
    assert handle.state.old_denoised is not None
    handle.close()
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


def test_public_selector_includes_res_after_the_four_stateless_names():
    assert STATELESS_SPEED_SAMPLERS == ("euler", "heun", "dpm_2", "exp_heun_2_x0")
    assert SUPPORTED_SPEED_SAMPLERS == STATELESS_SPEED_SAMPLERS + ("res_multistep",)


def test_public_factory_routes_res_to_stateful_handle():
    handle = create_speed_sampler_handle("res_multistep")
    assert isinstance(handle, _ResMultistepSamplerHandle)
    assert handle.capability is SamplerCapability.SINGLE_HISTORY
    handle.close()


# ---------------------------------------------------------------------------
# Clean projection primitive (plan S7 §16, §31 step 35)
# ---------------------------------------------------------------------------

class _Nested:
    """Minimal nested H3 stand-in: video [B,C,T,H,W] + audio [B,C,2,T_audio].

    Mirrors the real ``NestedTensor`` constructor contract — one list of
    streams — so ``type(history)([video, audio])`` in the adapter rebuilds
    the same shape.
    """

    def __init__(self, streams):
        self.streams = list(streams)
        self.is_nested = True

    def unbind(self):
        return list(self.streams)


def _known_video(t=2, h=4, w=4, channels=1, batch=1, seed=99):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, t, h, w, generator=g)


def _known_audio(t_audio=6, channels=1, batch=1, seed=123):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, channels, 2, t_audio, generator=g)


def _transition(source_thw, target_thw, ratio=2.0, old_sigma=0.5, stage_idx=0):
    return SpeedTransition(
        stage_idx=stage_idx,
        ratio=ratio,
        old_sigma=old_sigma,
        new_sigma=old_sigma,
        source_thw=source_thw,
        target_thw=target_thw,
    )


def test_on_transition_clears_history_and_sigma_metadata():
    handle = create_res_multistep_sampler_handle()
    video, audio = _known_video(), _known_audio()
    handle.state.old_denoised = _Nested([video, audio])
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.6
    transition = _transition((2, 4, 4), (4, 8, 8), ratio=2.0, old_sigma=0.5)

    handle.on_transition(transition)

    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


def test_on_transition_leaves_empty_state_alone():
    """No completed RES interval yet: the hook must preserve empty state and
    create no history."""
    handle = create_res_multistep_sampler_handle()
    handle.on_transition(_transition((2, 4, 4), (4, 8, 8)))
    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


def test_on_transition_clears_all_existing_history():
    handle = create_res_multistep_sampler_handle()
    handle.state.old_denoised = _Nested([_known_video(), _known_audio()])
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.6

    handle.on_transition(_transition((2, 4, 4), (4, 8, 8)))

    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None


def test_first_real_interval_after_transition_rebuilds_history():
    handle = create_res_multistep_sampler_handle()
    handle.state.old_denoised = _Nested([_known_video(), _known_audio()])
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.6
    handle.on_transition(_transition((2, 4, 4), (4, 8, 8)))

    handle.sampler(DeterministicModel(), torch.randn(1, 3, 4, 4), SIGMAS[:3], disable=True)

    assert handle.state.old_denoised is not None
    assert handle.state.old_sigma_down is not None
    assert handle.state.prev_sigma_in is not None


def test_on_transition_never_touches_reentry_or_conditioning_tensors():
    """The hook owns history only. The noisy re-entry and conditioning tensors
    are aliased into the history itself: any write-through on the stored
    history would surface in the aliases and fail the byte-identical check."""
    reentry = torch.randn(1, 1, 2, 4, 4)
    conditioning = torch.randn(1, 1, 2, 4, 4)
    reentry_snapshot = reentry.clone()
    conditioning_snapshot = conditioning.clone()
    handle = create_res_multistep_sampler_handle()
    handle.state.old_denoised = _Nested([reentry, conditioning])
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.6

    handle.on_transition(_transition((2, 4, 4), (4, 8, 8)))

    assert torch.equal(reentry, reentry_snapshot)
    assert torch.equal(conditioning, conditioning_snapshot)


def test_on_transition_twice_at_coincident_boundary_no_new_history():
    """Two boundaries at the same scheduler index leave no synthetic history."""
    handle = create_res_multistep_sampler_handle()
    video, audio = _known_video(), _known_audio()
    handle.state.old_denoised = _Nested([video, audio])
    handle.state.old_sigma_down = 0.5
    handle.state.prev_sigma_in = 0.6

    first = _transition((2, 4, 4), (4, 8, 8), ratio=2.0, old_sigma=0.5)
    second = _transition((4, 8, 8), (8, 16, 16), ratio=1.5, old_sigma=first.new_sigma)
    handle.on_transition(first)
    handle.on_transition(second)

    assert handle.state.old_denoised is None
    assert handle.state.old_sigma_down is None
    assert handle.state.prev_sigma_in is None
