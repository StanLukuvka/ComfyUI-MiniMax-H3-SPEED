"""Runtime observer contract for the multi-stage SPEED pipeline.

`run_speed_pipeline` notifies an optional observer about what the run does:
one event per actual denoising step, one event per resolution transition
(coincident transitions included), and run start/end markers. The observer
decides what to measure; this module imports nothing from the
spectral-analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeedStepEvent:
    """One actual denoising interval inside one SPEED stage.

    `actual_sigma` / `actual_sigma_next` come from the working sigma
    schedule the stage is consuming, so at a kappa-aligned re-entry they
    differ from `original_sigma` / `original_sigma_next`, which index the
    user's input schedule at the same global position.
    """

    callback_index: int
    stage_index: int
    stage_scale: float
    stage_local_step: int

    global_schedule_index: int

    actual_sigma: float
    actual_sigma_next: float

    original_sigma: float
    original_sigma_next: float

    stage_h: int
    stage_w: int
    stage_t: int

    full_h: int
    full_w: int
    full_t: int


@dataclass(frozen=True)
class SpeedTransitionEvent:
    """One resolution transition, including coincident ones.

    `sigma_before_alignment` is the working-schedule boundary value this
    transition read (already aligned if a coincident earlier transition
    patched it); `sigma_after_alignment` is the value written back.
    """

    transition_index: int

    from_stage: int
    to_stage: int

    global_schedule_index: int

    from_scale: float
    to_scale: float
    scale_ratio: float

    sigma_before_alignment: float
    sigma_after_alignment: float
    kappa: float

    source_h: int
    source_w: int

    target_h: int
    target_w: int


@dataclass(frozen=True)
class SpeedRunStartEvent:
    """Fired once before the first stage runs."""

    n_stages: int
    scales: tuple[float, ...]
    transition_steps: tuple[int, ...]
    global_steps: int
    full_h: int
    full_w: int
    full_t: int


@dataclass(frozen=True)
class SpeedRunEndEvent:
    """Fired once after the final stage completes."""

    n_stages: int
    global_steps: int
    transition_count: int


class SpeedRuntimeObserver(Protocol):
    """Minimal observer interface implemented by SPEED telemetry collectors."""

    def on_run_start(self, event: SpeedRunStartEvent) -> None: ...
    def on_step(self, event: SpeedStepEvent, x0, x) -> None: ...
    def on_transition(self, event: SpeedTransitionEvent) -> None: ...
    def on_run_end(self, event: SpeedRunEndEvent) -> None: ...


__all__ = [
    "SpeedRunEndEvent",
    "SpeedRunStartEvent",
    "SpeedStepEvent",
    "SpeedTransitionEvent",
    "SpeedRuntimeObserver",
]
