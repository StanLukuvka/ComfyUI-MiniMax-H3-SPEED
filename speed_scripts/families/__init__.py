"""Model families SPEED can run, picked with the nodes' ``model_type`` switch.

A family says how its latent splits into the spatial stream SPEED resizes
(always ``[B, C, T, H, W]`` here) and any extra stream it carries alongside,
how to read a stage's output back into the sampler's state, and how the extra
stream crosses a resolution change.

- ``h3_video``: MiniMax-H3 nested video/audio latent. Audio is carried on its
  own sigma clock.
- ``image``: a plain ``[B, C, H, W]`` or ``[B, C, 1, H, W]`` latent from a flow
  image model (Krea 2, MOSS Image, ...). No extra stream.

This module holds the family registry (``MODEL_TYPES``, ``get_family``,
``check_model_type``, ``prepare_latent``, ``main_stream``) and re-exports the
public names of the family modules: ``h3_video.py`` and ``image.py``.
"""

from __future__ import annotations

import torch

from .h3_video import H3VideoFamily, pack_h3, resolve_sigma_shifts, unpack_h3
from .image import ImageFamily, require_flow_sampling

_FAMILIES = {"h3_video": H3VideoFamily, "image": ImageFamily}
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


def main_stream(value, model_type):
    """Best-effort ``[B, C, T, H, W]`` spatial stream for diagnostics, or ``None``."""
    if model_type == "h3_video":
        # Only the nested video/audio latent is H3; a plain 5-D tensor (e.g. Krea 2) is not.
        if not getattr(value, "is_nested", False):
            return None
        value = next((s for s in value.unbind() if getattr(s, "ndim", 0) == 5), None)
        return value if isinstance(value, torch.Tensor) else None
    if not isinstance(value, torch.Tensor) or getattr(value, "is_nested", False):
        return None
    if value.ndim == 4:
        return value.unsqueeze(2)
    return value if value.ndim == 5 else None


__all__ = [
    "MODEL_TYPES",
    "H3VideoFamily",
    "ImageFamily",
    "get_family",
    "main_stream",
    "pack_h3",
    "prepare_latent",
    "require_flow_sampling",
    "resolve_sigma_shifts",
    "unpack_h3",
]
