"""Compatibility re-exports for the pre-consolidation Automatic planner module."""

from .planning import (
    PRESET_TO_STAGES,
    STAGES_TO_SCALES,
    build_automatic_speed_config,
)

__all__ = [
    "STAGES_TO_SCALES",
    "PRESET_TO_STAGES",
    "build_automatic_speed_config",
]
