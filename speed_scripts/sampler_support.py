"""Create the sampler used by SPEED.

Most samplers use ComfyUI directly. RES needs extra per-generation state, so
its wrapper also clears RES history whenever SPEED changes resolution.
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
    """How much state a sampler keeps between steps."""

    STATELESS_STEP_LOCAL = "stateless_step_local"
    SINGLE_HISTORY = "single_history"


@dataclass(frozen=True)
class SpeedTransition:
    """Details about one SPEED resolution change."""

    stage_idx: int
    ratio: float
    old_sigma: float
    new_sigma: float
    source_thw: tuple[int, int, int]
    target_thw: tuple[int, int, int]


class SpeedSamplerHandle:
    """Sampler object plus any state needed for one generation."""

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
    """RES sampler that clears its history at each resolution change."""

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
    """Create the sampler for this generation and reject unsupported names."""
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
