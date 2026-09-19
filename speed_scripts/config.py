"""Validated runtime configuration for the MiniMax-H3 SPEED sampler."""

from __future__ import annotations

from dataclasses import dataclass


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
    full_latent_h: int = 45
    full_latent_w: int = 80
    # Empty means full temporal resolution at every stage.
    temporal_scales: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        scales = tuple(float(s) for s in self.scales)
        steps = tuple(int(s) for s in self.transition_steps)
        if len(scales) < 1:
            raise ValueError("at least one scale required")
        if len(scales) == 1:
            if abs(scales[0] - 1.0) > 1e-6:
                raise ValueError("a single scale must be 1.0 (full resolution)")
            if steps:
                raise ValueError("single-scale config takes no transition steps")
        else:
            if abs(scales[-1] - 1.0) > 1e-6:
                raise ValueError("final scale must be 1.0 (full resolution)")
            if not all(0.0 < s <= 1.0 for s in scales):
                raise ValueError("every scale must be in (0, 1]")
            if not all(left < right for left, right in zip(scales[:-1], scales[1:])):
                raise ValueError("scales must be strictly increasing")
            if len(steps) != len(scales) - 1:
                raise ValueError("need (n_scales - 1) transition steps")
            if not all(s >= 1 for s in steps):
                raise ValueError("every transition step must be at least one")
            # Explicit user schedules must be strictly increasing. delta_custom
            # boundaries may later quantize onto the same sigma index.
            if self.transition_mode == "explicit" and any(
                a >= b for a, b in zip(steps[:-1], steps[1:])
            ):
                raise ValueError(
                    f"transition steps must be strictly increasing: got {list(steps)}"
                )
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.transition_mode not in ("explicit", "delta_custom"):
            raise ValueError("transition_mode must be 'explicit' or 'delta_custom'")
        if self.noise_amplitude <= 0.0 or self.noise_decay_exponent <= 0.0:
            raise ValueError("power spectrum A and beta must be positive")
        if self.full_latent_h < 1 or self.full_latent_w < 1:
            raise ValueError("full latent dims must be positive")
        if self.noise_policy not in NOISE_POLICIES:
            raise ValueError(f"unsupported noise_policy: {self.noise_policy}")
        if self.audio_policy not in AUDIO_POLICIES:
            raise ValueError(f"unsupported audio_policy: {self.audio_policy}")
        if self.sigma_policy not in SIGMA_POLICIES:
            raise ValueError(f"unsupported sigma_policy: {self.sigma_policy}")
        if self.audio_policy == "untouched" and self.sigma_policy != "no_alignment":
            raise ValueError("untouched audio requires sigma_policy=no_alignment")
        if self.temporal_scales:
            if len(self.temporal_scales) != len(scales):
                raise ValueError("temporal_scales must have the same length as scales")
            if not all(0.0 < s <= 1.0 for s in self.temporal_scales):
                raise ValueError("temporal scales must be in (0, 1]")
            if not all(
                left <= right
                for left, right in zip(self.temporal_scales[:-1], self.temporal_scales[1:])
            ):
                raise ValueError("temporal_scales must be non-decreasing")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "transition_steps", steps)


__all__ = [
    "NOISE_POLICIES",
    "AUDIO_POLICIES",
    "SIGMA_POLICIES",
    "RATIO_MODES",
    "SpeedConfig",
]
