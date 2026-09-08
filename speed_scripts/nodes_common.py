"""Small helpers shared by sampler node wrappers."""

from __future__ import annotations

from .h3_runtime import unpack_latent


def validate_transition_steps(transition_steps, n_sigmas):
    """Validate explicit boundaries against the shared-boundary slicing model.

    Boundaries are global sigma-schedule indices. Strictly increasing interior
    boundaries are sufficient: adjacent stages share the boundary sigma, so no
    extra ``n_stages * 2`` minimum-sigma heuristic is needed.
    """
    total_steps = n_sigmas - 1
    if any(not (0 < step < total_steps) for step in transition_steps):
        raise ValueError(
            f"transition goals must be interior step indices "
            f"(0 < goal < {total_steps}): got {list(transition_steps)}"
        )
    if any(left >= right for left, right in zip(transition_steps, transition_steps[1:])):
        raise ValueError(
            f"transition goals must be strictly increasing: got {list(transition_steps)}"
        )


def full_res_dims(latent_image) -> tuple[int, int]:
    """Resolve live full-resolution video H/W through the runtime unpacker."""
    samples = latent_image["samples"] if isinstance(latent_image, dict) else latent_image
    full_video, _ = unpack_latent(samples)
    return int(full_video.shape[-2]), int(full_video.shape[-1])


__all__ = ["validate_transition_steps", "full_res_dims"]
