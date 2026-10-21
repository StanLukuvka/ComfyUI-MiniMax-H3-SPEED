"""Model families SPEED can run, picked with the nodes' ``model_type`` switch.

A family says how its latent splits into the spatial stream SPEED resizes
(always ``[B, C, T, H, W]`` here) and any extra stream it carries alongside,
how to read a stage's output back into the sampler's state, and how the extra
stream crosses a resolution change.

- ``h3_video``: MiniMax-H3 nested video/audio latent. Audio is carried on its
  own sigma clock.
- ``image``: a plain ``[B, C, H, W]`` or ``[B, C, 1, H, W]`` latent from a flow
  image model (Krea 2, MOSS Image, ...). No extra stream.
"""

from __future__ import annotations

import torch

from .flow import (
    carry_preserved_audio,
    clock_reindex_audio_state,
    time_shift_sigma,
    to_internal_state,
)

def _model(guider):
    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    if model is None:
        raise ValueError("no model_patcher.model on guider")
    return model


def unpack_h3(samples):
    """Validate and unpack the H3 nested latent into ``(video, audio)``."""
    if not getattr(samples, "is_nested", False):
        raise ValueError(
            "model_type h3_video needs a MiniMax-H3 NestedTensor video/audio latent; "
            "use model_type image for image models"
        )
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError("MiniMax-H3 SPEED requires exactly video and audio streams")
    video, audio = streams
    if video.ndim != 5 or audio.ndim != 4:
        raise ValueError("expected H3 video [B,C,T,H,W] and audio [B,C,2,T]")
    if video.shape[0] != 1 or audio.shape[0] != 1:
        raise ValueError("MiniMax-H3 supports batch size one")
    if audio.shape[2] != 2:
        raise ValueError("MiniMax-H3 audio latent requires stereo axis size two")
    return video, audio


def pack_h3(video, audio):
    """Pack video/audio streams into ComfyUI's nested latent container."""
    from comfy import nested_tensor

    return nested_tensor.NestedTensor([video, audio])


def resolve_sigma_shifts(guider):
    """Read the H3 video/audio sigma shifts from the active model."""
    patcher = getattr(guider, "model_patcher", None)
    model = _model(guider)

    candidates = []
    model_options = getattr(guider, "model_options", None)
    if not isinstance(model_options, dict):
        model_options = getattr(patcher, "model_options", None)
    if isinstance(model_options, dict):
        transformer_options = model_options.get("transformer_options", {})
        if isinstance(transformer_options, dict):
            candidates.append((
                transformer_options.get("minimax_h3_sigma_shift_video"),
                transformer_options.get("minimax_h3_sigma_shift_audio"),
            ))

    candidates.append((
        getattr(model, "sigma_shift_video", None),
        getattr(model, "sigma_shift_audio", None),
    ))
    diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is not None:
        candidates.append((
            getattr(diffusion_model, "sigma_shift_video", None),
            getattr(diffusion_model, "sigma_shift_audio", None),
        ))

    shifts = next(
        (pair for pair in candidates if all(isinstance(value, (int, float)) for value in pair)),
        None,
    )
    if shifts is None:
        raise ValueError(
            "active MiniMax-H3 sigma shifts are unavailable: expected "
            "sigma_shift_video/audio on the H3 model or transformer options"
        )
    video_shift, audio_shift = map(float, shifts)
    if video_shift <= 0.0 or audio_shift <= 0.0:
        raise ValueError("active MiniMax-H3 shifts must be positive")
    return video_shift, audio_shift, video_shift / audio_shift


class H3VideoFamily:
    """MiniMax-H3: resize the video stream, carry audio on its own sigma clock."""

    name = "h3_video"

    def __init__(self, guider):
        self.video_shift, self.audio_shift, self.audio_scale = resolve_sigma_shifts(guider)

    def split(self, samples):
        return unpack_h3(samples)

    def join(self, main, extra):
        return pack_h3(main, extra)

    def to_internal(self, public, sigma):
        # The H3 latent format is the identity, so the output is already in model space.
        video, audio = self.split(public)
        return to_internal_state(video, audio, sigma, self.audio_scale)

    def carry_extra(self, audio, old_sigma, new_sigma, policy, x0):
        old_audio_sigma = time_shift_sigma(old_sigma, self.video_shift, self.audio_shift)
        new_audio_sigma = time_shift_sigma(new_sigma, self.video_shift, self.audio_shift)
        if policy == "carry_preserve":
            return carry_preserved_audio(audio, old_sigma, new_sigma, old_audio_sigma, new_audio_sigma)
        if policy == "clock_reindex":
            if x0 is None:
                raise RuntimeError("clock_reindex requires an x0 callback from this stage")
            _, clean_audio = self.split(x0)
            return clock_reindex_audio_state(
                audio, clean_audio, old_sigma, new_sigma,
                old_audio_sigma, new_audio_sigma, self.audio_scale,
            )
        return audio


def _sampling_name(sampling):
    """Name the comfy.model_sampling bases of a ModelSampling object, e.g. ``ModelSamplingDiscrete / EPS``."""
    bases = [
        cls.__name__
        for cls in type(sampling).__mro__[1:]
        if cls.__module__ == "comfy.model_sampling"
    ]
    return " / ".join(bases) or type(sampling).__name__


def require_flow_sampling(guider):
    """Raise unless the guider's model samples as a flow (CONST) model.

    Image-to-image flow variants are rejected too: they condition on a source image.
    """
    import comfy.model_sampling

    sampling = guider.model_patcher.get_model_object("model_sampling")
    if not isinstance(sampling, comfy.model_sampling.CONST) or isinstance(
        sampling,
        (comfy.model_sampling.IMG_TO_IMG_FLOW, comfy.model_sampling.IMG_TO_IMG_VELOCITY),
    ):
        raise ValueError(
            "model_type image supports flow (CONST) models only; "
            f"this model uses {_sampling_name(sampling)}"
        )


class ImageFamily:
    """Flow image models: one spatial latent, resized on its own.

    The latent's rank follows the model's latent format: ``[B, C, H, W]`` for
    2-dimensional formats, ``[B, C, 1, H, W]`` for 3-dimensional ones such as
    Wan 2.1 (Krea 2), which ComfyUI's ``fix_empty_latent_channels`` unsqueezes.
    """

    name = "image"

    def __init__(self, guider):
        require_flow_sampling(guider)
        self.model = _model(guider)
        latent_format = guider.model_patcher.get_model_object("latent_format")
        self.frame_axis = getattr(latent_format, "latent_dimensions", 2) == 3

    def split(self, samples):
        if getattr(samples, "is_nested", False):
            raise ValueError("model_type image got a nested video/audio latent; use model_type h3_video")
        single_frame = samples.ndim == 5 and samples.shape[2] == 1
        if samples.ndim == 4 and not self.frame_axis:
            return samples.unsqueeze(2), None
        if single_frame and self.frame_axis:
            return samples, None
        if single_frame:
            raise ValueError(
                "model_type image got a [B,C,1,H,W] latent, but this model's latent format "
                "is 2-dimensional and takes [B,C,H,W]; use an image empty-latent node"
            )
        expected = "[B,C,1,H,W]" if self.frame_axis else "[B,C,H,W]"
        raise ValueError(
            f"model_type image expects a {expected} latent for this model; got {list(samples.shape)}"
        )

    def join(self, main, extra):
        return main if self.frame_axis else main.squeeze(2)

    def to_internal(self, public, sigma):
        # ComfyUI hands back process_latent_out(x / (1 - sigma)); undo both to get x.
        # Image latent formats can be affine (Krea 2 uses Wan 2.1's mean/std).
        main, _ = self.split(self.model.process_latent_in(public))
        return main * (1.0 - sigma), None

    def carry_extra(self, extra, old_sigma, new_sigma, policy, x0):
        return None


_FAMILIES = {"h3_video": H3VideoFamily, "image": ImageFamily}
MODEL_TYPES = tuple(_FAMILIES)


def prepare_latent(guider, latent):
    """Copy of ``latent`` with the empty latent fitted to the model, as SamplerCustomAdvanced does.

    Generic empty-latent nodes (e.g. EmptyLatentImage) give 4 channels; ComfyUI widens
    them to the model's latent format. Nested (H3) latents pass through unchanged.
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
        if getattr(value, "is_nested", False):
            value = next((s for s in value.unbind() if getattr(s, "ndim", 0) == 5), None)
        return value if isinstance(value, torch.Tensor) and value.ndim == 5 else None
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
