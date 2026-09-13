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
only talk to the handle. All solver math stays inside ComfyUI's native
sampler objects — this repository never reimplements a solver.
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

#: Public selector list. PR B adds ``res_multistep`` here only after its
#: state-preservation tests and real H3 validation pass.
SUPPORTED_SPEED_SAMPLERS = STATELESS_SPEED_SAMPLERS


class SamplerCapability(Enum):
    """What a sampler needs from the SPEED stage loop."""

    #: Pure per-step function; no state survives a stage boundary.
    STATELESS_STEP_LOCAL = "stateless_step_local"
    #: Keeps step history across stage boundaries; the handle preserves it.
    SINGLE_HISTORY = "single_history"


@dataclass(frozen=True)
class SpeedTransition:
    """One configured SPEED transition boundary.

    Produced by the stage loop after the boundary sigma has been aligned and
    patched into the working schedule, and handed to the sampler handle's
    transition hook exactly once.

    ``source_stream_shapes`` lists the per-stream latent shapes the stage's
    sampler saw, in the host's flat-pack order (video first, then audio).
    Stateless samplers ignore it; the stateful RES handle needs it to slice
    a flat packed history tensor back into its video and audio streams,
    because the host packs nested latents flat before any sampler code runs.
    """

    stage_idx: int
    ratio: float
    old_sigma: float
    new_sigma: float
    source_thw: tuple[int, int, int]
    target_thw: tuple[int, int, int]
    source_stream_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None


class SpeedSamplerHandle:
    """Run-scoped wrapper around one SPEED sampler choice.

    The stage loop receives this handle from ``run_speed_pipeline`` and never
    touches the raw sampler name or any sampler state. Stateless samplers
    need nothing at boundaries, so both methods are no-ops; stateful handles
    (PR B ``res_multistep``) override them. The handle never mutates the
    scheduler or the working sigma schedule.
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
    object keeps that state across every stage's ``guider.sample()`` call,
    so RES history survives stage boundaries; ``close()`` (run-level
    cleanup) releases it. Never routes through the stage-resetting native
    ``sampler_object("res_multistep")``.
    """

    def __init__(self):
        from .res_multistep_adapter import ResMultistepSampler, ResMultistepState

        self.state = ResMultistepState()
        self.sampler = ResMultistepSampler(self.state)
        self.capability = SamplerCapability.SINGLE_HISTORY

    def on_transition(self, transition: SpeedTransition) -> None:
        from .res_multistep_adapter import (
            project_clean_history,
            rebase_res_history_sigmas,
        )

        # No completed RES interval yet: nothing to project or rebase. The
        # empty state is preserved as-is; no history is created here.
        if self.state.old_denoised is None:
            return
        # Clean-history projection: replace the stored video geometry with the
        # clean spectral projection; the clean audio estimate passes through
        # unchanged. On the real host path the history is the flat packed
        # tensor the guider produced at the sampler boundary, so the
        # per-stream shapes from that pack ride along for the video/audio
        # split. Sigma metadata moves to the aligned next-stage coordinate
        # system. Coincident boundaries simply run this again on the already
        # projected history; new history is never synthesized here.
        self.state.old_denoised = project_clean_history(
            self.state.old_denoised,
            transition.target_thw,
            source_stream_shapes=transition.source_stream_shapes,
        )
        rebase_res_history_sigmas(
            self.state, transition.new_sigma, transition.ratio
        )
        # Deliberately untouched: the scheduler and working sigma schedule,
        # the noisy re-entry tensors, and all H3 conditioning tensors.

    def close(self) -> None:
        self.state.clear()


def create_speed_sampler_handle(sampler_name: str) -> SpeedSamplerHandle:
    """Build the run-scoped handle for ``sampler_name``.

    Unknown names raise ``ValueError`` listing the invalid name and the
    supported names. Never falls back to Euler.
    """
    if sampler_name not in SUPPORTED_SPEED_SAMPLERS:
        supported = ", ".join(repr(name) for name in SUPPORTED_SPEED_SAMPLERS)
        raise ValueError(
            f"Unsupported SPEED sampler {sampler_name!r}. "
            f"Supported samplers: {supported}."
        )
    return _StatelessSamplerHandle(sampler_name)


def create_res_multistep_sampler_handle() -> "_ResMultistepSamplerHandle":
    """Build the run-scoped stateful RES handle (runtime-only seam).

    Not part of the public selector yet: ``res_multistep`` joins
    ``SUPPORTED_SPEED_SAMPLERS`` only after the RES state-preservation
    tests and real H3 validation pass (plan PR B acceptance). The runtime
    reaches RES through this factory alone.
    """
    return _ResMultistepSamplerHandle()
