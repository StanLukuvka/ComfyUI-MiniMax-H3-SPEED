"""Shared tail for the two sampler nodes (preset- and manual-schedule variants).

Both nodes end the same way: validate the transition schedule against the
sigma count, build a SpeedConfig from the live latent dims, and run the
multi-stage SPEED chain. Consolidating that here keeps the twin `sample()`
bodies from drifting apart — one change to the pipeline call contract lands
once, not twice.
"""

from __future__ import annotations

import comfy.samplers

from .config import SpeedConfig
from .h3_runtime import run_speed_pipeline, unpack_latent


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


def build_config_and_run(
    noise, guider, sigmas, latent_image, *,
    scales, transition_steps, transition_mode,
    noise_policy, delta, noise_amplitude, noise_decay_exponent, seed_offset,
):
    """Validate, build the SpeedConfig from the live latent, run SPEED."""
    n_stages = len(scales)
    if n_stages < 2:
        raise ValueError("need at least two stages (scales ending at 1.0)")
    validate_transition_steps(transition_steps, n_stages, len(sigmas))

    full_video, _ = unpack_latent(latent_image.get("samples"))
    config = SpeedConfig(
        scales=tuple(scales),
        transition_steps=tuple(transition_steps),
        transition_mode=transition_mode,
        noise_policy=noise_policy,
        delta=float(delta),
        noise_amplitude=float(noise_amplitude),
        noise_decay_exponent=float(noise_decay_exponent),
        transition_seed_offset=int(seed_offset),
        full_latent_h=int(full_video.shape[-2]),
        full_latent_w=int(full_video.shape[-1]),
    )

    # Run the multi-stage SPEED diffusion chain. Audio is carried through
    # unchanged; the final-stage x0 (denoised) is surfaced as denoised_output.
    # NOTE: SPEED's kappa alignment and DCT-boundary semantics are calibrated
    # for Euler. Supporting other samplers requires calibration and changing
    # the underlying math for kappa alignment.
    return run_speed_pipeline(
        noise,
        guider,
        sigmas,
        latent_image,
        config,
        sampler=comfy.samplers.sampler_object("euler"),
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        output_device=None,
    )


__all__ = ["validate_transition_steps", "build_config_and_run"]
