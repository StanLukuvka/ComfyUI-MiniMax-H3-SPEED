"""Shared tail for the two sampler nodes (preset- and manual-schedule variants).

The Manual node validates its transition schedule against the sigma count
through `validate_transition_steps` here (the Automatic node's boundaries
come from the delta-threshold resolver, which the runtime already bounds),
and both nodes resolve the full-res latent dims the same way (via the same
`unpack_latent` the SPEED pipeline uses). The build-and-run tail is inlined
into each node's sample() method — the LatentWalker + SpeedConfig wiring is
short, and one function per node keeps the wiring obvious.
"""

from __future__ import annotations

from .h3_runtime import unpack_latent


def validate_transition_steps(transition_steps, n_stages, n_sigmas):
    """Fail fast with stage-indexed messages if the schedule cannot fit.

    Shared contract with SpeedConfig (count, >= 1) and run_speed_pipeline
    (interior): boundaries must be strictly increasing interior step indices
    of the sigma schedule.

    Those two checks are also the FULL length requirement — no separate
    minimum-sigma rule exists. Derivation from the runtime slicing
    (`run_speed_pipeline` slices stage k over working_sigmas[prev..ts_k],
    boundary indices shared, aligned re-entry replacing the boundary entry):
    stage 0 covers [0, ts0] with ts0 >= 1 (>= 1 step), each interior stage
    [ts(k-1), tsk] with ts(k-1) < tsk (>= 1 step), and the final stage
    [ts(last), n_sigmas - 1] has >= 1 step exactly when ts(last) <=
    n_sigmas - 2 — which is precisely the interior check. So a schedule
    passing here always runs every stage with at least one denoising step;
    the minimum sigma count for a given boundary set is max(ts) + 2.

    `n_stages` is accepted for call-signature symmetry with SpeedConfig
    (which validates len(steps) == n_stages - 1) but is not part of the
    length requirement: adjacent SPEED stages share the boundary sigma, so
    stages do NOT need two unique sigmas each.
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


def full_res_dims(latent_image) -> tuple[int, int]:
    """Resolve (full_latent_h, full_latent_w) from a ComfyUI LATENT dict.

    Reuses the SPEED pipeline's own `unpack_latent` so the H/W validation
    (and any future geometry checks) is consistent between the node's
    SpeedConfig and the runtime's first call.
    """
    samples = latent_image["samples"] if isinstance(latent_image, dict) else latent_image
    full_video, _ = unpack_latent(samples)
    return int(full_video.shape[-2]), int(full_video.shape[-1])


__all__ = ["validate_transition_steps", "full_res_dims"]
