"""Flow (CONST) image models (``flow_image``): one spatial latent, no extra stream."""

from __future__ import annotations

import torch


def _model(guider):
    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    if model is None:
        raise ValueError("no model_patcher.model on guider")
    return model


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
            "model_type flow_image supports flow (CONST) models only; "
            f"this model uses {_sampling_name(sampling)}"
        )


class FlowImageFamily:
    """Flow image models: one spatial latent, resized on its own.

    The latent's rank follows the model's latent format: ``[B, C, H, W]`` for
    2-dimensional formats, ``[B, C, 1, H, W]`` for 3-dimensional ones such as
    Wan 2.1 (Krea 2), which ComfyUI's ``fix_empty_latent_channels`` unsqueezes.
    """

    name = "flow_image"
    # Image-to-image: a VAE-encoded latent seeds the first stage, as SamplerCustomAdvanced does.
    accepts_source_latent = True

    def __init__(self, guider):
        require_flow_sampling(guider)
        self.model = _model(guider)
        latent_format = guider.model_patcher.get_model_object("latent_format")
        self.frame_axis = getattr(latent_format, "latent_dimensions", 2) == 3

    def split(self, samples):
        if getattr(samples, "is_nested", False):
            raise ValueError("model_type flow_image got a nested video/audio latent; use model_type h3_video")
        single_frame = samples.ndim == 5 and samples.shape[2] == 1
        if samples.ndim == 4 and not self.frame_axis:
            return samples.unsqueeze(2), None
        if single_frame and self.frame_axis:
            return samples, None
        if single_frame:
            raise ValueError(
                "model_type flow_image got a [B,C,1,H,W] latent, but this model's latent format "
                "is 2-dimensional and takes [B,C,H,W]; use an image empty-latent node"
            )
        expected = "[B,C,1,H,W]" if self.frame_axis else "[B,C,H,W]"
        raise ValueError(
            f"model_type flow_image expects a {expected} latent for this model; got {list(samples.shape)}"
        )

    def join(self, main, extra):
        return main if self.frame_axis else main.squeeze(2)

    def main_stream(self, value):
        """Best-effort ``[B, C, T, H, W]`` spatial stream for diagnostics, or ``None``."""
        if not isinstance(value, torch.Tensor) or getattr(value, "is_nested", False):
            return None
        if value.ndim == 4:
            return value.unsqueeze(2)
        return value if value.ndim == 5 else None

    def to_internal(self, public, sigma):
        # ComfyUI hands back process_latent_out(x / (1 - sigma)); undo both to get x.
        # Image latent formats can be affine (Krea 2 uses Wan 2.1's mean/std).
        main, _ = self.split(self.model.process_latent_in(public))
        return main * (1.0 - sigma), None

    def carry_extra(self, extra, old_sigma, new_sigma, policy, x0):
        return None
