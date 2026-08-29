"""Shared tail for the two sampler nodes (preset- and manual-schedule variants).

Both nodes validate their transition schedule against the sigma count
through the same helper here. The build-and-run tail is inlined into each
node's sample() method (LatentWalker + SpeedConfig are short, and one
function per node keeps the wiring obvious).
"""

from __future__ import annotations


def validate_transition_steps(transition_steps, n_stages, n_sigmas):
    """Fail fast with stage-indexed messages if the schedule cannot fit.

    Shared contract with SpeedConfig (count, >= 1) and run_speed_pipeline
    (interior): boundaries must be strictly increasing interior step indices
    of the sigma schedule, and each stage needs at least two sigmas with the
    last boundary leaving room for the final stage's slice.
    """
    total_steps = n_sigmas - 1
    if any(not (0 < ts < total_steps) for ts in transition_steps):
        raise ValueError(
            f"transition goals must be interior step indices (0 < goal < "
            f"{total_steps}): got {list(transition_steps)}"
        )
    if any(a >= b for a, b in zip(transition_steps[:-1], transition_steps[1:])):
        raise ValueError(
            f"transition goals must be strictly increasing: got {list(transition_steps)}"
        )
    # Need at least 2 sigmas per stage, plus enough room for transition steps.
    # max(transition_steps) is the last boundary index, so we need that + 1
    # to cover the final stage's sigma slice.
    min_required = max(n_stages * 2, max(transition_steps) + 1)
    if n_sigmas < min_required:
        raise ValueError(
            f"sigma schedule too short: got {n_sigmas} sigmas, need "
            f"at least {min_required} with transition steps {list(transition_steps)}. "
            f"Increase steps to >= {min_required - 1}."
        )


__all__ = ["validate_transition_steps"]
