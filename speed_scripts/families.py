"""MiniMax-H3 latent helpers: split, pack and read the sigma shifts of the nested video/audio latent."""


def unpack_latent(samples):
    """Validate and unpack the H3 nested latent into ``(video, audio)``."""
    if not getattr(samples, "is_nested", False):
        raise ValueError("MiniMax-H3 SPEED requires a NestedTensor video/audio latent")
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


def pack_latent(video, audio):
    """Pack video/audio streams into ComfyUI's nested latent container."""
    from comfy import nested_tensor

    return nested_tensor.NestedTensor([video, audio])


def resolve_sigma_shifts(guider):
    """Read the H3 video/audio sigma shifts from the active model."""
    patcher = getattr(guider, "model_patcher", None)
    model = getattr(patcher, "model", None)
    if model is None:
        raise ValueError("no model_patcher.model on guider")

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
