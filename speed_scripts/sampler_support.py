"""Sampler-support layer for the SPEED multi-stage pipeline.

The SPEED stage loop in ``h3_runtime`` runs one sampler through several
progressive-resolution stages. This module owns everything sampler-related
for that loop:

* the public, curated list of sampler names SPEED supports,
* a capability classification per sampler,
* a run-scoped ``SpeedSamplerHandle`` that wraps the sampler object plus any
  run-scoped state, with a transition hook called once per configured SPEED
  transition and a ``close()`` for end-of-run cleanup.

The stage loop and scheduler logic never branch on the sampler name; they
only talk to the handle. Stateless samplers use ComfyUI's native sampler
objects. RES uses this repository's deterministic stateful adapter because
its run-scoped state and boundary policy belong to the sampler handle. The
SPEED scheduler itself remains sampler-agnostic.

RES boundary modes are explicit policies. ``reset`` is the default and clears
history. ``projected`` is the historical project-and-rebase comparison path.
``hybrid`` is experimental V3.0: it prepares one spatial-only video DCT
candidate mix, keeps audio first-order, and falls back to reset when temporal
geometry changes. Flat host tensors require recorded stream shapes. Missing or
inconsistent metadata must raise rather than guess a video/audio split. These
paths have automated tests but no native ComfyUI or GPU validation claim.
"""


from dataclasses import dataclass
from enum import Enum


#: Stateless, step-local samplers exposed by SPEED. Their behavior at a
#: SPEED stage boundary needs no cross-stage state, so their handle hook is
#: a no-op.
STATELESS_SPEED_SAMPLERS = (
    "euler",
    "heun",
    "dpm_2",
    "exp_heun_2_x0",
)

#: Public selector list. RES is stateful and uses the adapter below.
#: All five names are exposed by the three public nodes.
SUPPORTED_SPEED_SAMPLERS = STATELESS_SPEED_SAMPLERS + (
    "res_multistep",
)

RES_HISTORY_MODES = ("reset", "projected", "hybrid")


class SamplerCapability(Enum):
    """What a sampler needs from the SPEED stage loop."""

    #: Pure per-step function; no state survives a stage boundary.
    STATELESS_STEP_LOCAL = "stateless_step_local"
    #: Owns one previous-step history record; the run-scoped handle decides
    #: how that history is treated at a SPEED boundary.
    SINGLE_HISTORY = "single_history"


@dataclass(frozen=True)
class SpeedTransition:
    """One configured SPEED transition boundary.

    Produced by the stage loop after the boundary sigma has been aligned and
    patched into the working schedule, and handed to the sampler handle's
    transition hook exactly once.

    ``source_stream_shapes`` lists the per-stream latent shapes the source
    stage's sampler saw, in the host's flat-pack order (video first, then
    audio). Stateless samplers ignore it. RES ``projected`` mode uses it to
    split a flat packed history tensor back into its video and audio streams;
    ``reset`` mode does not need it. ``target_stream_shapes`` records the
    authoritative post-transition pack layout for hybrid boundary metadata.
    Hybrid rejects flat input when either shape record is missing or does not
    match the packed tensor, so it cannot silently mix the wrong streams.
    """

    stage_idx: int
    ratio: float
    old_sigma: float
    new_sigma: float
    source_thw: tuple[int, int, int]
    target_thw: tuple[int, int, int]
    source_stream_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    target_stream_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None


class SpeedSamplerHandle:
    """Run-scoped wrapper around one SPEED sampler choice.

    The stage loop receives this handle from ``run_speed_pipeline`` and never
    touches the raw sampler name or any sampler state. Stateless samplers
    need nothing at boundaries, so both methods are no-ops; stateful
    ``res_multistep`` handles override them. The handle owns its boundary
    policy and never mutates the scheduler or working sigma schedule.
    """

    sampler: object
    capability: SamplerCapability

    def on_transition(self, transition: SpeedTransition) -> None:
        pass

    def close(self) -> None:
        pass


class _StatelessSamplerHandle(SpeedSamplerHandle):
    """Run-scoped handle for a stateless, step-local sampler."""

    def __init__(self, name: str):
        # Built exactly once per SPEED run, here — never per stage, never
        # per denoising step. Imported lazily because the surrounding
        # package must import (and its tests must collect) outside a full
        # ComfyUI install, which only provides comfy.samplers at runtime.
        from comfy.samplers import sampler_object

        self.sampler = sampler_object(name)
        self.capability = SamplerCapability.STATELESS_STEP_LOCAL


class _ResMultistepSamplerHandle(SpeedSamplerHandle):
    """Run-scoped handle for the stateful RES Multistep adapter.

    Owns one ``ResMultistepState`` per SPEED run. The wrapped sampler
    object keeps that state across each stage's ``guider.sample()`` call, and
    the handle decides what survives a stage boundary; ``close()`` (run-level
    cleanup) releases it. Never routes through the stage-resetting native
    ``sampler_object("res_multistep")``.
    """

    def __init__(self, history_mode: str = "reset"):
        from .res_multistep_adapter import ResMultistepSampler, ResMultistepState

        if history_mode not in RES_HISTORY_MODES:
            raise ValueError(
                f"Unsupported RES history mode {history_mode!r}. "
                f"Supported modes: {', '.join(repr(mode) for mode in RES_HISTORY_MODES)}."
            )

        self.history_mode = history_mode
        self.state = ResMultistepState()
        self.sampler = ResMultistepSampler(self.state)
        self.capability = SamplerCapability.SINGLE_HISTORY

    def on_transition(self, transition: SpeedTransition) -> None:
        if self.history_mode == "reset":
            self.state.clear()
            return

        if self.history_mode == "projected":
            if self.state.old_denoised is None:
                return
            from .res_multistep_adapter import (
                project_clean_history,
                rebase_res_history_sigmas,
            )

            self.state.old_denoised = project_clean_history(
                self.state.old_denoised,
                transition.target_thw,
                source_stream_shapes=transition.source_stream_shapes,
            )
            rebase_res_history_sigmas(
                self.state, transition.new_sigma, transition.ratio
            )
            return

        if self.history_mode == "hybrid":
            if self.state.old_denoised is None:
                self.state.clear()
                return
            if transition.source_thw[0] != transition.target_thw[0]:
                # V3 only mixes spatial boundaries. Temporal changes fall back
                # to reset until a temporal hybrid operator is defined.
                self.state.clear()
                return

            from .res_multistep_adapter import (
                project_clean_history,
                rebase_res_history_sigmas,
            )

            if not self.state.hybrid_pending:
                self.state.hybrid_second_order_thw = transition.source_thw
            source_stream_shapes = (
                transition.source_stream_shapes
                or self.state.hybrid_target_stream_shapes
            )
            self.state.old_denoised = project_clean_history(
                self.state.old_denoised,
                transition.target_thw,
                source_stream_shapes=source_stream_shapes,
            )
            rebase_res_history_sigmas(
                self.state, transition.new_sigma, transition.ratio
            )
            self.state.hybrid_target_stream_shapes = transition.target_stream_shapes
            self.state.hybrid_pending = True
            return

        # Construction validates the mode, so reaching this branch means the
        # handle was mutated or a future mode was added without an explicit
        # boundary implementation. Never silently treat a new mode as
        # ``projected``.
        raise RuntimeError(f"Unhandled RES history mode {self.history_mode!r}")

    def close(self) -> None:
        self.state.clear()


def create_speed_sampler_handle(
    sampler_name: str,
    *,
    res_history_mode: str = "reset",
) -> SpeedSamplerHandle:
    """Build the run-scoped handle for ``sampler_name``.

    Unknown names raise ``ValueError`` listing the invalid name and the
    supported names. Never falls back to Euler.
    """
    if res_history_mode not in RES_HISTORY_MODES:
        raise ValueError(
            f"Unsupported RES history mode {res_history_mode!r}. "
            f"Supported modes: {', '.join(repr(mode) for mode in RES_HISTORY_MODES)}."
        )
    if sampler_name not in SUPPORTED_SPEED_SAMPLERS:
        supported = ", ".join(repr(name) for name in SUPPORTED_SPEED_SAMPLERS)
        raise ValueError(
            f"Unsupported SPEED sampler {sampler_name!r}. "
            f"Supported samplers: {supported}."
        )
    if sampler_name == "res_multistep":
        return create_res_multistep_sampler_handle(history_mode=res_history_mode)

    return _StatelessSamplerHandle(sampler_name)


def create_res_multistep_sampler_handle(
    *,
    history_mode: str = "reset",
) -> "_ResMultistepSamplerHandle":
    """Build the run-scoped stateful RES handle (runtime-only seam).

    The public selector routes ``res_multistep`` through this factory so the
    runtime keeps one stateful handle for the whole SPEED run.
    """
    return _ResMultistepSamplerHandle(history_mode=history_mode)
