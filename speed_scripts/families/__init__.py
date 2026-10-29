"""Model families SPEED can run, picked with the nodes' ``model_type`` switch.

A family splits its latent into the spatial stream SPEED resizes (always
``[B, C, T, H, W]`` here) and an optional extra stream, reads a stage's output
back into sampler state, and carries the extra stream across a resolution
change. This module holds the registry and re-exports the family modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .h3_video import H3VideoFamily, pack_h3, resolve_sigma_shifts, unpack_h3
from .flow_image import FlowImageFamily, require_flow_sampling

_FAMILIES = {"h3_video": H3VideoFamily, "flow_image": FlowImageFamily}
MODEL_TYPES = tuple(_FAMILIES)


def prepare_latent(guider, latent):
    """Copy of ``latent`` with the empty latent fitted to the model, as SamplerCustomAdvanced does.

    Generic empty-latent nodes (e.g. EmptyLatentImage) give 4 channels; ComfyUI widens
    them to the model's latent format. Nested (h3_video) latents pass through unchanged.
    """
    import comfy.sample

    latent = latent.copy()
    latent["samples"] = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    return latent


def check_model_type(model_type):
    """Raise unless ``model_type`` is one of the supported switch values."""
    if model_type not in _FAMILIES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}; got {model_type!r}")
    return model_type


def get_family(model_type, guider):
    """Build the family for the node's ``model_type`` switch."""
    return _FAMILIES[check_model_type(model_type)](guider)


@dataclass(frozen=True)
class PreparedLatent:
    """The family plus the fitted latent dict and its ``main``/``extra`` streams (``extra`` may be ``None``)."""

    family: H3VideoFamily | FlowImageFamily
    latent: dict
    main: torch.Tensor
    extra: torch.Tensor | None


def prepare_family(model_type, guider, latent) -> PreparedLatent:
    """Pick the family, fit the latent and split it, checking it is unmasked.

    Only a family with ``accepts_source_latent`` takes a non-empty latent (image-to-image).
    """
    if "noise_mask" in latent:
        raise ValueError("SPEED does not support noise masks")
    family = get_family(model_type, guider)
    latent = prepare_latent(guider, latent)

    main, extra = family.split(latent.get("samples"))
    empty = not torch.count_nonzero(main) and (extra is None or not torch.count_nonzero(extra))
    if not empty and not family.accepts_source_latent:
        raise ValueError(f"model_type {family.name} requires an empty input latent")
    return PreparedLatent(family=family, latent=latent, main=main, extra=extra)


__all__ = [
    "MODEL_TYPES",
    "H3VideoFamily",
    "FlowImageFamily",
    "get_family",
    "pack_h3",
    "PreparedLatent",
    "prepare_family",
    "prepare_latent",
    "require_flow_sampling",
    "resolve_sigma_shifts",
    "unpack_h3",
]
