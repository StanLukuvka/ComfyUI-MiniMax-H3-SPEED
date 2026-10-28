"""Transition-coordinate math for SPEED stage changes.

``aligned_sigma`` maps a sigma across a resolution change. ``reentry_noise``
rescales carried state back to noise at a stage's entry sigma.
"""

from __future__ import annotations


def aligned_sigma(sigma: float, resolution_ratio: float) -> tuple[float, float]:
    q = float(sigma)
    ratio = float(resolution_ratio)
    if not 0.0 < q < 1.0:
        raise ValueError("transition sigma must be between zero and one")
    if ratio <= 1.0:
        raise ValueError("resolution ratio must be greater than one")
    kappa = ratio / (1.0 + (ratio - 1.0) * q)
    return kappa, q * kappa


def reentry_noise(internal_state, start_sigma: float):
    if start_sigma <= 0.0:
        raise ValueError("start_sigma must be positive")
    return internal_state / start_sigma
