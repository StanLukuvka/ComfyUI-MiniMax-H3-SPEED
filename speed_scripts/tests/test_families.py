"""Model families: the model_type switch, H3 sigma shifts, and the image family."""

import re

import pytest
import torch

from conftest import (
    SIGMAS,
    EchoGuider,
    OracleFlowGuider,
    Patcher,
    SeededNoise,
    StubSampler,
    flow_patcher,
    image_latent,
    make_latent,
    model_sampling,
)
from speed_scripts.config import SpeedConfig
from speed_scripts.families import ImageFamily, get_family, main_stream, resolve_sigma_shifts
from speed_scripts.runtime import run_speed_pipeline

CFG = SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), transition_mode="explicit")


def _guider(patcher):
    return type("Guider", (), {"model_patcher": patcher})()


# ---------------------------------------------------------------------------
# model_type switch
# ---------------------------------------------------------------------------

def test_model_type_must_match_the_latent():
    guider = OracleFlowGuider(torch.zeros(1, 4, 8, 8))
    with pytest.raises(ValueError, match="use model_type h3_video"):
        run_speed_pipeline(SeededNoise(7), guider, SIGMAS, make_latent(), CFG,
                           sampler=StubSampler(), model_type="image")
    with pytest.raises(ValueError, match="use model_type image"):
        run_speed_pipeline(SeededNoise(7), EchoGuider(), SIGMAS, image_latent((1, 4, 8, 8)), CFG,
                           sampler=StubSampler(), model_type="h3_video")
    with pytest.raises(ValueError, match="model_type must be one of"):
        get_family("sdxl", guider)


def test_main_stream_by_model_type():
    image = torch.zeros(1, 4, 8, 8)
    assert main_stream(image, "image").shape == (1, 4, 1, 8, 8)
    assert main_stream(image, "h3_video") is None
    assert main_stream(make_latent()["samples"], "image") is None
    assert main_stream(make_latent()["samples"], "h3_video").ndim == 5


# ---------------------------------------------------------------------------
# h3_video
# ---------------------------------------------------------------------------

def test_h3_sigma_shifts_win_over_generic_comfy_sampling_shift():
    generic = type("Sampling", (), {"shift": 1.0, "audio_shift": 1.0})()
    model = type("Model", (), {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0})()
    video, audio, scale = resolve_sigma_shifts(_guider(Patcher(model, model_sampling=generic)))
    assert (video, audio, scale) == pytest.approx((12.0, 3.0, 4.0))


def test_non_h3_model_is_rejected_instead_of_using_generic_shift():
    generic = type("Sampling", (), {"shift": 1.0, "audio_shift": 1.0})()
    with pytest.raises(ValueError, match="MiniMax-H3 sigma shifts are unavailable"):
        resolve_sigma_shifts(_guider(Patcher(object(), model_sampling=generic)))


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "base,name",
    [
        ("EPS", "ModelSamplingDiscrete / EPS"),
        ("IMG_TO_IMG_FLOW", "ModelSamplingDiscrete / IMG_TO_IMG_FLOW / CONST"),
        ("IMG_TO_IMG_VELOCITY", "ModelSamplingDiscrete / IMG_TO_IMG_VELOCITY / CONST"),
    ],
)
def test_image_family_rejects_non_flow_sampling(base, name):
    import comfy.model_sampling as ms

    guider = _guider(flow_patcher(model_sampling(ms.ModelSamplingDiscrete, getattr(ms, base))))
    with pytest.raises(ValueError, match=f"flow \\(CONST\\) models only; this model uses {re.escape(name)}$"):
        get_family("image", guider)


# FLOW_AV models build ``class ModelSampling(ModelSamplingAV, CONST)`` in model_base.py.
@pytest.mark.parametrize("first", ["ModelSamplingDiscrete", "ModelSamplingAV"])
def test_image_family_accepts_flow_sampling(first):
    import comfy.model_sampling as ms

    guider = _guider(flow_patcher(model_sampling(getattr(ms, first), ms.CONST)))
    assert isinstance(get_family("image", guider), ImageFamily)


def test_image_family_matches_the_latent_rank_to_the_latent_format():
    flat = ImageFamily(_guider(flow_patcher(latent_dimensions=2)))
    with pytest.raises(ValueError, match=r"got a \[B,C,1,H,W\] latent, but this model's latent format is 2-dimensional"):
        flat.split(torch.zeros(1, 4, 1, 8, 8))

    framed = ImageFamily(_guider(flow_patcher(latent_dimensions=3)))
    main, _ = framed.split(torch.zeros(1, 4, 1, 8, 8))
    assert framed.join(main, None).shape == (1, 4, 1, 8, 8)
    with pytest.raises(ValueError, match=r"expects a \[B,C,1,H,W\] latent for this model"):
        framed.split(torch.zeros(1, 4, 8, 8))


def test_image_family_rejects_multi_frame_latents():
    family = ImageFamily(_guider(flow_patcher()))
    with pytest.raises(ValueError, match=r"expects a \[B,C,H,W\] latent for this model; got \[1, 4, 3, 8, 8\]"):
        family.split(torch.zeros(1, 4, 3, 8, 8))
