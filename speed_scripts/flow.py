"""Flow-coordinate helpers for MiniMax-H3 SPEED transitions.

Functions are dependency-free and intentionally accept tensor-like values where
ordinary multiplication and division are defined.
"""

from __future__ import annotations


def aligned_sigma(sigma: float, resolution_ratio: float) -> tuple[float, float]:
    as_q = float(sigma)
    ratio = float(resolution_ratio)
    if not 0.0 < as_q < 1.0:
        raise ValueError("transition sigma must be between zero and one")
    if ratio <= 1.0:
        raise ValueError("resolution ratio must be greater than one")
    kappa = ratio / (1.0 + (ratio - 1.0) * as_q)
    return kappa, as_q * kappa


def time_shift_sigma(sigma, from_shift: float, to_shift: float):
    if from_shift <= 0.0 or to_shift <= 0.0:
        raise ValueError("sigma shifts must be positive")
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def to_internal_state(video_public, audio_public, sigma: float, audio_scale: float):
    if not 0.0 <= sigma < 1.0:
        raise ValueError("endpoint sigma must be in [0, 1)")
    if audio_scale <= 0.0:
        raise ValueError("audio_scale must be positive")
    clean_weight = 1.0 - sigma
    return video_public * clean_weight, audio_public * audio_scale * clean_weight


def carry_preserved_audio(
    carried_audio,
    old_video_sigma: float,
    new_video_sigma: float,
    old_audio_sigma: float,
    new_audio_sigma: float,
):
    values = (old_video_sigma, new_video_sigma, old_audio_sigma, new_audio_sigma)
    if any(value <= 0.0 for value in values):
        raise ValueError("carry conversion sigmas must be positive")
    return carried_audio * (new_video_sigma / new_audio_sigma) * (
        old_audio_sigma / old_video_sigma
    )


def clock_reindex_audio_state(
    carried_audio,
    clean_carried_audio,
    old_video_sigma: float,
    new_video_sigma: float,
    old_audio_sigma: float,
    new_audio_sigma: float,
    audio_scale: float,
):
    values = (old_video_sigma, new_video_sigma, old_audio_sigma, new_audio_sigma, audio_scale)
    if any(value <= 0.0 for value in values):
        raise ValueError("clock re-indexing values must be positive")
    current_native = carried_audio * old_audio_sigma / old_video_sigma
    clean_native = clean_carried_audio / audio_scale
    noise_native = (
        current_native - (1.0 - old_audio_sigma) * clean_native
    ) / old_audio_sigma
    new_native = (
        (1.0 - new_audio_sigma) * clean_native
        + new_audio_sigma * noise_native
    )
    return new_native * new_video_sigma / new_audio_sigma


def reentry_noise(internal_state, start_sigma: float):
    if start_sigma <= 0.0:
        raise ValueError("start_sigma must be positive")
    return internal_state / start_sigma
