"""RES Multistep adapter: state model, deterministic pipeline, and the
same-resolution split oracle (plan S7 §11-§15).

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

Everything here runs against a deterministic fake model on plain float
schedules — no ComfyUI import is needed below the handle seam.
"""

import torch
import pytest

from speed_scripts.res_multistep_adapter import (
    ResMultistepSampler,
    ResMultistepState,
    res_multistep_sampler,
)
from speed_scripts.sampler_support import (
    STATELESS_SPEED_SAMPLERS,
    SUPPORTED_SPEED_SAMPLERS,
    SamplerCapability,
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

def test_state_starts_empty_and_clear_releases_everything():
    state = ResMultistepState()
    assert state.old_denoised is None
    assert state.old_sigma_down is None
    assert state.prev_sigma_in is None
    tensor = torch.zeros(2, 2)
    state.old_denoised = tensor
    state.old_sigma_down = 0.5
    state.prev_sigma_in = 0.9
    state.clear()
    assert state.old_denoised is None
    assert state.old_sigma_down is None
    assert state.prev_sigma_in is None


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
# Handle seam (plan S7 §5/§13; public selector stays PR A)
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


def test_public_selector_still_excludes_res_multistep():
    assert SUPPORTED_SPEED_SAMPLERS == STATELESS_SPEED_SAMPLERS
    assert "res_multistep" not in SUPPORTED_SPEED_SAMPLERS


def test_public_factory_still_rejects_res_multistep_fail_closed():
    with pytest.raises(ValueError) as excinfo:
        create_speed_sampler_handle("res_multistep")
    assert "res_multistep" in str(excinfo.value)
