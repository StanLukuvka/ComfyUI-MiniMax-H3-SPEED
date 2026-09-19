"""Compatibility re-exports for the pre-consolidation Automatic planner module."""

from .planning import STAGES_TO_SCALES, build_automatic_speed_config

__all__ = [
    "STAGES_TO_SCALES",
    "build_automatic_speed_config",
]
