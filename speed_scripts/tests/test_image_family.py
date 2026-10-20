"""model_type=image: plain image latents through the SPEED runtime."""

import re

import pytest
import torch

from conftest import StubSampler, make_latent
from speed_scripts.config import SpeedConfig
from speed_scripts.families import ImageFamily, get_family, main_stream
from speed_scripts.flow import aligned_sigma
from speed_scripts.runtime import run_speed_pipeline
from speed_scripts.spectral import lowpass_dct

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])
CFG = SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), transition_mode="explicit")


class AffineModel:
    """Latent format with a mean/std like Wan 2.1's (used by Krea 2)."""

    mean, std = 0.7, 2.0

    def process_latent_in(self, x):
        return (x - self.mean) / self.std

    def process_latent_out(self, x):
        return x * self.std + self.mean


def _flow_patcher(model, sampling=None):
    """Patcher stand-in: ``get_model_object`` returns the latent sampling class, as in ComfyUI."""
    import comfy.model_sampling

    sampling = comfy.model_sampling.CONST() if sampling is None else sampling
    return type("P", (), {
        "model": model,
        "get_model_object": lambda self, name: sampling,
    })()


class OracleFlowGuider:
    """Mimics ComfyUI's CFGGuider.sample for a CONST flow model with an ideal denoiser.

    The zero latent is not shifted, the start state is ``sigma0 * noise``, the
    denoiser always predicts ``clean`` (low-passed to the stage size), and the
    result goes through ``inverse_noise_scaling`` and ``process_latent_out``.
    """

    def __init__(self, clean):
        self.model_patcher = _flow_patcher(AffineModel())
        self.clean = clean
        self.starts = []
        self.ends = []
        self.original_conds = None

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        assert not torch.count_nonzero(latent_image)
        s0, s1 = float(sigmas[0]), float(sigmas[-1])
        x0 = s0 * noise
        clean = lowpass_dct(self.clean, tuple(noise.shape[-2:])).reshape(noise.shape)
        x1 = clean + (s1 / s0) * (x0 - clean)
        self.starts.append(x0)
        self.ends.append(x1)
        if callback is not None:
            callback(0, clean, x1, len(sigmas) - 1)
        out = x1 / (1.0 - s1) if s1 < 1.0 else x1
        return self.model_patcher.model.process_latent_out(out)


class ImageNoise:
    seed = 7

    def generate_noise(self, latent):
        shape = latent["samples"].shape
        return torch.randn(shape, generator=torch.Generator().manual_seed(self.seed))


def _image_latent(shape):
    return {"samples": torch.zeros(shape)}


@pytest.mark.parametrize("shape", [(2, 4, 8, 8), (1, 4, 1, 8, 8)])
def test_image_run_keeps_shape_and_carries_state_across_the_affine_latent_format(shape):
    clean = torch.randn((shape[0], 4, 8, 8), generator=torch.Generator().manual_seed(1))
    guider = OracleFlowGuider(clean)
    out, denoised = run_speed_pipeline(
        ImageNoise(), guider, SIGMAS, _image_latent(shape), CFG,
        sampler=StubSampler(), model_type="image", disable_pbar=True,
    )
    assert out["samples"].shape == shape
    assert denoised["samples"].shape == shape
    assert [tuple(s.shape[-2:]) for s in guider.starts] == [(4, 4), (8, 8)]

    # The second stage starts from the first stage's end state, grown in DCT space and
    # rescaled by kappa: its low band is exactly kappa * that state.
    boundary = float(SIGMAS[5])
    kappa, _ = aligned_sigma(boundary, 2.0)
    carried = lowpass_dct(guider.starts[1], (4, 4))
    assert torch.allclose(carried, kappa * guider.ends[0], atol=1e-5)

    # An ideal denoiser lands on the clean latent; the output is in latent-format space.
    expected = AffineModel().process_latent_out(clean).reshape(shape)
    assert torch.allclose(out["samples"], expected, atol=1e-5)


def test_coupled_full_grid_runs_for_image_latents():
    shape = (1, 4, 8, 8)
    guider = OracleFlowGuider(torch.zeros(shape))
    cfg = SpeedConfig(scales=(.5, 1.0), transition_steps=(5,), transition_mode="explicit",
                      noise_policy="coupled_full_grid")
    out, _ = run_speed_pipeline(
        ImageNoise(), guider, SIGMAS, _image_latent(shape), cfg,
        sampler=StubSampler(), model_type="image", disable_pbar=True,
    )
    assert out["samples"].shape == shape


def test_model_type_must_match_the_latent():
    guider = OracleFlowGuider(torch.zeros(1, 4, 8, 8))
    with pytest.raises(ValueError, match="use model_type h3_video"):
        run_speed_pipeline(ImageNoise(), guider, SIGMAS, make_latent(), CFG,
                           sampler=StubSampler(), model_type="image")
    with pytest.raises(ValueError, match="use model_type image"):
        run_speed_pipeline(ImageNoise(), _h3_guider(), SIGMAS, _image_latent((1, 4, 8, 8)), CFG,
                           sampler=StubSampler(), model_type="h3_video")
    with pytest.raises(ValueError, match="model_type must be one of"):
        get_family("sdxl", guider)


def _real_shaped_sampling(base):
    """Same shape as ComfyUI's ``class ModelSampling(s, c)`` in model_base.py."""
    import comfy.model_sampling

    return type("ModelSampling", (comfy.model_sampling.ModelSamplingDiscrete, base), {})()


def test_image_family_rejects_non_flow_sampling():
    import comfy.model_sampling

    cases = (
        (comfy.model_sampling.EPS, "ModelSamplingDiscrete / EPS"),
        (comfy.model_sampling.IMG_TO_IMG_FLOW, "ModelSamplingDiscrete / IMG_TO_IMG_FLOW / CONST"),
        (comfy.model_sampling.IMG_TO_IMG_VELOCITY, "ModelSamplingDiscrete / IMG_TO_IMG_VELOCITY / CONST"),
    )
    for base, expected in cases:
        guider = type("G", (), {"model_patcher": _flow_patcher(AffineModel(), _real_shaped_sampling(base))})()
        with pytest.raises(ValueError, match=f"flow \\(CONST\\) models only; this model uses {re.escape(expected)}$"):
            get_family("image", guider)
    ok = type("G", (), {"model_patcher": _flow_patcher(AffineModel(), _real_shaped_sampling(comfy.model_sampling.CONST))})()
    assert isinstance(get_family("image", ok), ImageFamily)


def test_image_family_rejects_multi_frame_latents():
    family = ImageFamily(OracleFlowGuider(torch.zeros(1)))
    with pytest.raises(ValueError, match=r"\[B,C,H,W\] or \[B,C,1,H,W\]"):
        family.split(torch.zeros(1, 4, 3, 8, 8))


def test_main_stream_by_model_type():
    image = torch.zeros(1, 4, 8, 8)
    assert main_stream(image, "image").shape == (1, 4, 1, 8, 8)
    assert main_stream(image, "h3_video") is None
    assert main_stream(make_latent()["samples"], "image") is None
    assert main_stream(make_latent()["samples"], "h3_video").ndim == 5


def _h3_guider():
    model = type("M", (), {"sigma_shift_video": 12.0, "sigma_shift_audio": 3.0})()
    return type("G", (), {"model_patcher": type("P", (), {"model": model})()})()


def test_runtime_fits_a_generic_empty_latent_to_the_model_first(monkeypatch):
    import comfy.sample

    seen = []

    def widen(model, latent_image, *ratios):
        seen.append(tuple(latent_image.shape))
        return torch.zeros(latent_image.shape[0], 4, *latent_image.shape[2:])

    monkeypatch.setattr(comfy.sample, "fix_empty_latent_channels", widen)
    guider = OracleFlowGuider(torch.zeros(1, 4, 8, 8))
    out, _ = run_speed_pipeline(
        ImageNoise(), guider, SIGMAS, _image_latent((1, 1, 8, 8)), CFG,
        sampler=StubSampler(), model_type="image", disable_pbar=True,
    )
    assert seen == [(1, 1, 8, 8)]
    assert out["samples"].shape == (1, 4, 8, 8)
