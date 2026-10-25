"""MiniMax-H3 (``h3_video``) model family.

The H3 latent is a nested video/audio pair: the video stream is the spatial
stream SPEED resizes, and the audio stream is carried across resolution
changes on its own sigma clock. This module holds the family class together
with its unpack/pack helpers, the sigma-shift resolution, and the H3-only
audio-state math moved from ``speed_scripts/flow.py``.
"""

from __future__ import annotations


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
            "use model_type flow_image for image models"
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
