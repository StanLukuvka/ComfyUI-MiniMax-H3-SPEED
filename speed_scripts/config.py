"""Validated runtime configuration for the MiniMax-H3 SPEED sampler."""

from __future__ import annotations

from dataclasses import dataclass
import math


NOISE_POLICIES = {"direct_coarse", "coupled_full_grid"}
AUDIO_POLICIES = {"clock_reindex", "carry_preserve", "untouched"}
SIGMA_POLICIES = {"canonical", "no_alignment"}
RATIO_MODES = ("steps", "ratio")


@dataclass(frozen=True)
class SpeedConfig:
    """Multi-stage progressive-resolution SPEED configuration."""

    scales: tuple[float, ...] = (0.5, 1.0)
    transition_steps: tuple[int, ...] = (5,)
    noise_policy: str = "direct_coarse"
    audio_policy: str = "clock_reindex"
    sigma_policy: str = "canonical"
    transition_seed_offset: int = 10_000
    transition_mode: str = "explicit"
    delta: float = 0.01
    noise_amplitude: float = 12.105
    noise_decay_exponent: float = 0.773
    # Optional temporal ladder; when provided it must end at full temporal resolution.
    temporal_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        scales = tuple(float(scale) for scale in self.scales)
        steps = tuple(int(step) for step in self.transition_steps)

        if self.transition_mode not in ("explicit", "delta_custom"):
            raise ValueError("transition_mode must be 'explicit' or 'delta_custom'")
        if len(scales) < 2:
            raise ValueError("at least two scales required")
        if not all(0.0 < scale <= 1.0 for scale in scales):
            raise ValueError("every scale must be in (0, 1]")
        if abs(scales[-1] - 1.0) > 1e-6:
            raise ValueError("final scale must be 1.0 (full resolution)")
        if not all(left < right for left, right in zip(scales[:-1], scales[1:])):
            raise ValueError("scales must be strictly increasing")

        if self.transition_mode == "explicit":
            if len(steps) != len(scales) - 1:
                raise ValueError("need (n_scales - 1) transition steps")
            if not all(step >= 1 for step in steps):
                raise ValueError("every transition step must be at least one")
            if any(left >= right for left, right in zip(steps[:-1], steps[1:])):
                raise ValueError(
                    f"transition steps must be strictly increasing: got {list(steps)}"
                )
        else:
            # delta_custom computes boundaries from the sigma schedule at runtime;
            # there is no reason to carry placeholder transition indices.
            steps = ()

        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if (
            not math.isfinite(self.noise_amplitude)
            or not math.isfinite(self.noise_decay_exponent)
            or self.noise_amplitude <= 0.0
            or self.noise_decay_exponent <= 0.0
        ):
            raise ValueError("power spectrum A and beta must be positive finite values")
        if self.noise_policy not in NOISE_POLICIES:
            raise ValueError(f"unsupported noise_policy: {self.noise_policy}")
        if self.audio_policy not in AUDIO_POLICIES:
            raise ValueError(f"unsupported audio_policy: {self.audio_policy}")
        if self.sigma_policy not in SIGMA_POLICIES:
            raise ValueError(f"unsupported sigma_policy: {self.sigma_policy}")
        if self.audio_policy == "untouched" and self.sigma_policy != "no_alignment":
            raise ValueError("untouched audio requires sigma_policy=no_alignment")

        temporal_scales = tuple(float(scale) for scale in self.temporal_scales)
        if temporal_scales:
            if len(temporal_scales) != len(scales):
                raise ValueError("temporal_scales must have the same length as scales")
            if not all(0.0 < scale <= 1.0 for scale in temporal_scales):
                raise ValueError("temporal scales must be in (0, 1]")
            if not all(left <= right for left, right in zip(temporal_scales[:-1], temporal_scales[1:])):
                raise ValueError("temporal_scales must be non-decreasing")
            if abs(temporal_scales[-1] - 1.0) > 1e-6:
                raise ValueError("final temporal scale must be 1.0 (full temporal resolution)")

        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "transition_steps", steps)
        object.__setattr__(self, "temporal_scales", temporal_scales)


__all__ = [
    "NOISE_POLICIES",
    "AUDIO_POLICIES",
    "SIGMA_POLICIES",
    "RATIO_MODES",
    "SpeedConfig",
]
