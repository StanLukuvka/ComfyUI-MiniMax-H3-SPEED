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

RES uses reset-only boundary behavior. It clears its previous-step history
at every SPEED stage boundary.
"""

from dataclasses import dataclass
from enum import Enum


STATELESS_SPEED_SAMPLERS = (
    "euler",
    "heun",
    "dpm_2",
    "exp_heun_2_x0",
)

SUPPORTED_SPEED_SAMPLERS = STATELESS_SPEED_SAMPLERS + ("res_multistep",)


class SamplerCapability(Enum):
    """What a sampler needs from the SPEED stage loop."""

    STATELESS_STEP_LOCAL = "stateless_step_local"
    SINGLE_HISTORY = "single_history"


@dataclass(frozen=True)
class SpeedTransition:
    """One configured SPEED transition boundary."""

    stage_idx: int
    ratio: float
    old_sigma: float
    new_sigma: float
    source_thw: tuple[int, int, int]
    target_thw: tuple[int, int, int]


class SpeedSamplerHandle:
    """Run-scoped wrapper around one SPEED sampler choice."""

    sampler: object
    capability: SamplerCapability

    def on_transition(self, transition: SpeedTransition) -> None:
        pass

    def close(self) -> None:
        pass


class _StatelessSamplerHandle(SpeedSamplerHandle):
    def __init__(self, name: str):
        from comfy.samplers import sampler_object

        self.sampler = sampler_object(name)
        self.capability = SamplerCapability.STATELESS_STEP_LOCAL


class _ResMultistepSamplerHandle(SpeedSamplerHandle):
    """Run-scoped deterministic RES sampler with reset-only boundaries."""

    def __init__(self):
        from .res_multistep_adapter import ResMultistepSampler, ResMultistepState

        self.state = ResMultistepState()
        self.sampler = ResMultistepSampler(self.state)
        self.capability = SamplerCapability.SINGLE_HISTORY

    def on_transition(self, transition: SpeedTransition) -> None:
        self.state.clear()

    def close(self) -> None:
        self.state.clear()


def create_speed_sampler_handle(sampler_name: str) -> SpeedSamplerHandle:
    """Build the run-scoped handle for ``sampler_name`` and fail closed."""
    if sampler_name not in SUPPORTED_SPEED_SAMPLERS:
        supported = ", ".join(repr(name) for name in SUPPORTED_SPEED_SAMPLERS)
        raise ValueError(
            f"Unsupported SPEED sampler {sampler_name!r}. "
            f"Supported samplers: {supported}."
        )
    if sampler_name == "res_multistep":
        return create_res_multistep_sampler_handle()
    return _StatelessSamplerHandle(sampler_name)


def create_res_multistep_sampler_handle() -> "_ResMultistepSamplerHandle":
    return _ResMultistepSamplerHandle()
