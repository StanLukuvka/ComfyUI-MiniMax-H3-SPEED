"""Validated configurations for the MiniMax-H3 SPEED sampler node.

Contains the SpeedConfig dataclass, scale presets, and default transition
steps. Ported from the Lab's speed_lab/config.py — minimal subset that
the MVP needs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


SCALE_PRESETS: dict[str, tuple[float, ...]] = {
    "half_then_full": (0.5, 1.0),
    "quarter_half_full": (0.25, 0.5, 1.0),
    "quarter_half_3q_full": (0.25, 0.5, 0.75, 1.0),
    "aggressive": (0.25, 0.75, 1.0),
    "three_quarter_then_full": (0.75, 1.0),
}

DEFAULT_TRANSITION_STEPS: dict[str, tuple[int, ...]] = {
    "half_then_full": (5,),
    "quarter_half_full": (3, 5),
    "quarter_half_3q_full": (3, 5, 8),
    "aggressive": (3, 8),
    "three_quarter_then_full": (10,),
}

NOISE_POLICIES = {"direct_coarse", "coupled_full_grid"}
AUDIO_POLICIES = {"clock_reindex", "carry_preserve", "untouched"}
SIGMA_POLICIES = {"canonical", "no_alignment"}


@dataclass(frozen=True)
class SpeedConfig:
    """Multi-stage progressive-resolution SPEED configuration."""

    scales: tuple[float, ...] = (0.5, 1.0)
    transition_steps: tuple[int, ...] = (5,)
    noise_policy: str = "direct_coarse"
    audio_policy: str = "clock_reindex"
    sigma_policy: str = "canonical"
    transition_seed_offset: int = 10_000
    transition_mode: str = "explicit"  # "explicit" uses transition_steps; "delta_custom" computes from power spectrum
    delta: float = 0.01
    # Initial estimates for MiniMax-H3 (not WAN 2.1). Replace after running
    # harvest calibration on a real H3 model — see harvest.py / SigmaHarvest node.
    power_A: float = 150.0
    power_beta: float = 2.0
    full_latent_h: int = 45
    full_latent_w: int = 80
    certification: str = "requires_h3_gpu_validation"
    # Temporal scale per stage (empty = no temporal scaling, run full T at every stage).
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
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.transition_mode not in ("explicit", "delta_custom"):
            raise ValueError("transition_mode must be 'explicit' or 'delta_custom'")
        if self.power_A <= 0.0 or self.power_beta <= 0.0:
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
            if not all(left <= right for left, right in zip(self.temporal_scales[:-1], self.temporal_scales[1:])):
                raise ValueError("temporal_scales must be non-decreasing")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "transition_steps", steps)

    @property
    def is_ablation(self) -> bool:
        return not (
            self.noise_policy == "direct_coarse"
            and self.audio_policy == "clock_reindex"
            and self.sigma_policy == "canonical"
        )

    @property
    def claims_canonical_speed(self) -> bool:
        return not self.is_ablation

    @property
    def n_stages(self) -> int:
        return len(self.scales)

    def with_overrides(self, **values) -> "SpeedConfig":
        return replace(self, **values)


def preset_config(preset: str, *, noise="direct_coarse", audio="clock_reindex",
                  sigma="canonical", seed_offset=10_000, delta=0.01,
                  mode="explicit") -> SpeedConfig:
    """Build a SpeedConfig from a named scale preset."""
    if preset not in SCALE_PRESETS:
        raise ValueError(f"unknown preset: {preset} (available: {sorted(SCALE_PRESETS)})")
    scales = SCALE_PRESETS[preset]
    steps = DEFAULT_TRANSITION_STEPS[preset]
    return SpeedConfig(
        scales=scales,
        transition_steps=steps,
        noise_policy=noise,
        audio_policy=audio,
        sigma_policy=sigma,
        transition_seed_offset=seed_offset,
        delta=delta,
        transition_mode=mode,
    )


def canonical_config() -> SpeedConfig:
    return preset_config("half_then_full")


def coupled_noise_config() -> SpeedConfig:
    return preset_config("half_then_full", noise="coupled_full_grid")


def carry_preserve_config() -> SpeedConfig:
    return preset_config("half_then_full", audio="carry_preserve")


def no_alignment_config() -> SpeedConfig:
    return preset_config("half_then_full", audio="untouched", sigma="no_alignment")


__all__ = ["SCALE_PRESETS", "DEFAULT_TRANSITION_STEPS", "SpeedConfig",
           "preset_config", "canonical_config", "coupled_noise_config",
           "carry_preserve_config", "no_alignment_config"]
