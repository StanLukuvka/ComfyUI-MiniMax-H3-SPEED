"""Model families SPEED can run, picked with the nodes' ``model_type`` switch.

A family says how its latent splits into the spatial stream SPEED resizes
(always ``[B, C, T, H, W]`` here) and any extra stream it carries alongside,
how to read a stage's output back into the sampler's state, and how the extra
stream crosses a resolution change.

- ``h3_video``: MiniMax-H3 nested video/audio latent. Audio is carried on its
  own sigma clock.
- ``flow_image``: a plain ``[B, C, H, W]`` or ``[B, C, 1, H, W]`` latent from a
  flow image model (Krea 2, MOSS Image, ...). No extra stream.

This module holds the family registry (``MODEL_TYPES``, ``get_family``,
``check_model_type``, ``prepare_latent``, ``prepare_family``) and re-exports
the public names of the family modules: ``h3_video.py`` and ``flow_image.py``.
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
    """One family's view of the input latent, ready for planning and running.

    ``latent`` is the dict after ``prepare_latent``; ``main`` and ``extra`` are
    the streams from ``family.split`` (``extra`` is ``None`` when the family
    has no extra stream).
    """

    family: H3VideoFamily | FlowImageFamily
    latent: dict
    main: torch.Tensor
    extra: torch.Tensor | None


def prepare_family(model_type, guider, latent) -> PreparedLatent:
    """Pick the family and split the empty latent, in the runtime's old order.

    Runs, in order: the noise-mask check, ``get_family``, ``prepare_latent``,
    ``family.split``, and the empty-latent check. Same error messages as the
    pre-split runtime.
    """
    if "noise_mask" in latent:
        raise ValueError("SPEED does not support noise masks")
    family = get_family(model_type, guider)
    latent = prepare_latent(guider, latent)

    main, extra = family.split(latent.get("samples"))
    if torch.count_nonzero(main) or (extra is not None and torch.count_nonzero(extra)):
        raise ValueError("SPEED requires an empty input latent")
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
