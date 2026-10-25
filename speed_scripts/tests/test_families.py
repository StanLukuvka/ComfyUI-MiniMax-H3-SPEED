"""Model families: the model_type switch and its errors, the image flow check and rank guard."""

import pytest
import torch

from conftest import (
    SIGMAS,
    EchoGuider,
    EchoNoise,
    OracleFlowGuider,
    Patcher,
    StubSampler,
    flow_patcher,
    image_latent,
    make_latent,
    model_sampling,
)
from speed_scripts.config import SpeedConfig
from speed_scripts.families import FlowImageFamily, get_family
from speed_scripts.runtime import run_speed_pipeline


def _guider(patcher):
    return type("Guider", (), {"model_patcher": patcher})()


def _image_guider(sampling=None, latent_dimensions=2):
    return _guider(flow_patcher(sampling, latent_dimensions))


@pytest.mark.parametrize(
    "model_type,guider,latent,error",
    [
        ("flow_image", OracleFlowGuider(torch.zeros(1)), make_latent(), "use model_type h3_video"),
        ("h3_video", EchoGuider(), image_latent((1, 4, 8, 8)), "use model_type flow_image"),
        ("sdxl", EchoGuider(), make_latent(), r"model_type must be one of \('h3_video', 'flow_image'\)"),
        # The image path runs flow (CONST) models only, image-to-image flow variants excluded.
        ("flow_image", _image_guider(model_sampling("ModelSamplingDiscrete", "EPS")), image_latent((1, 4, 8, 8)),
         r"flow \(CONST\) models only; this model uses ModelSamplingDiscrete / EPS$"),
        ("flow_image", _image_guider(model_sampling("ModelSamplingDiscrete", "IMG_TO_IMG_FLOW")), image_latent((1, 4, 8, 8)),
         r"this model uses ModelSamplingDiscrete / IMG_TO_IMG_FLOW / CONST$"),
        ("flow_image", _image_guider(model_sampling("ModelSamplingDiscrete", "IMG_TO_IMG_VELOCITY")), image_latent((1, 4, 8, 8)),
         r"this model uses ModelSamplingDiscrete / IMG_TO_IMG_VELOCITY / CONST$"),
        # The image latent's rank must match the model's latent format.
        ("flow_image", _image_guider(latent_dimensions=2), image_latent((1, 4, 1, 8, 8)),
         r"got a \[B,C,1,H,W\] latent, but this model's latent format is 2-dimensional"),
        ("flow_image", _image_guider(latent_dimensions=3), image_latent((1, 4, 8, 8)),
         r"expects a \[B,C,1,H,W\] latent for this model; got \[1, 4, 8, 8\]"),
        ("flow_image", _image_guider(), image_latent((1, 4, 3, 8, 8)),
         r"expects a \[B,C,H,W\] latent for this model; got \[1, 4, 3, 8, 8\]"),
        # H3 reads its sigma shifts from the model, never from a generic sampling shift.
        ("h3_video", _guider(Patcher(object(), model_sampling=type("S", (), {"shift": 1.0})())), make_latent(),
         "MiniMax-H3 sigma shifts are unavailable"),
    ],
    ids=["image_on_h3", "h3_on_image", "unknown", "eps", "img2img_flow", "img2img_velocity",
         "rank_2d_format", "rank_3d_format", "multi_frame", "non_h3_model"],
)
def test_model_type_errors_name_the_setting_to_use(model_type, guider, latent, error):
    cfg = SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), transition_mode="explicit")
    with pytest.raises(ValueError, match=error):
        run_speed_pipeline(EchoNoise(), guider, SIGMAS, latent, cfg,
                           sampler=StubSampler(), model_type=model_type)


# FLOW_AV models build ``class ModelSampling(ModelSamplingAV, CONST)`` in model_base.py.
@pytest.mark.parametrize("first_base", ["ModelSamplingDiscrete", "ModelSamplingAV"])
def test_image_family_accepts_flow_sampling(first_base):
    guider = _image_guider(model_sampling(first_base, "CONST"))
    assert isinstance(get_family("flow_image", guider), FlowImageFamily)
