"""Shared pytest bootstrap and ComfyUI fakes for the SPEED test suite."""

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
NODES_DIR = REPO_ROOT / "nodes"
for _p in (str(REPO_ROOT), str(NODES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def install_comfy_stubs():
    """Install the minimal comfy.* surface required by node/runtime tests."""
    comfy = ModuleType("comfy")
    samplers = ModuleType("comfy.samplers")
    utils = ModuleType("comfy.utils")
    model_mgmt = ModuleType("comfy.model_management")
    kdiff = ModuleType("comfy.k_diffusion")
    ksampling = ModuleType("comfy.k_diffusion.sampling")
    nested_tensor = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True

        def __init__(self, tensors):
            self._tensors = list(tensors)

        def unbind(self):
            return list(self._tensors)

    nested_tensor.NestedTensor = NestedTensor
    samplers.sampler_object = lambda name: ("sampler", name)
    utils.PROGRESS_BAR_ENABLED = True

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.node_id = node_id

        def update_absolute(self, value, total=None, preview=None):
            pass

        def update(self, value):
            pass

    utils.ProgressBar = ProgressBar

    def pack_latents(latents):
        shapes, tensors = [], []
        for tensor in latents:
            shapes.append(list(tensor.shape))
            tensors.append(tensor.reshape(tensor.shape[0], 1, -1))
        return torch.cat(tensors, dim=-1), shapes

    def unpack_latents(combined, shapes):
        out, work = [], combined
        for shape in shapes:
            cut = math.prod(shape[1:])
            out.append(work[:, :, :cut].reshape([work.shape[0]] + shape[1:]))
            work = work[:, :, cut:]
        return out

    utils.pack_latents = pack_latents
    utils.unpack_latents = unpack_latents
    model_mgmt.intermediate_device = lambda: "cpu"

    def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            denoised = model(x, sigma, **extra_args)
            d = (x - denoised) / sigma
            x = x + d * (sigmas[i + 1] - sigma)
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "denoised": denoised})
        return x

    ksampling.sample_euler = sample_euler

    comfy.samplers = samplers
    comfy.utils = utils
    comfy.model_management = model_mgmt
    comfy.k_diffusion = kdiff
    comfy.k_diffusion.sampling = ksampling
    comfy.nested_tensor = nested_tensor
    sys.modules["comfy"] = comfy
    for name, mod in (
        ("samplers", samplers),
        ("utils", utils),
        ("model_management", model_mgmt),
        ("k_diffusion", kdiff),
        ("k_diffusion.sampling", ksampling),
        ("nested_tensor", nested_tensor),
    ):
        sys.modules[f"comfy.{name}"] = mod


def make_nested(video, audio):
    return type(
        "Nested",
        (),
        {"is_nested": True, "unbind": lambda self: [video, audio]},
    )()


def make_latent(*, h=8, w=8, t=2, channels=1, **metadata):
    video = torch.zeros(1, channels, t, h, w)
    audio = torch.zeros(1, 1, 2, 44)
    latent = {"samples": make_nested(video, audio)}
    latent.update(metadata)
    return latent


def make_fake_noise(seed=42, calls=None):
    class FakeNoise:
        def __init__(self):
            self.seed = seed

        def generate_noise(self, latent):
            if calls is not None:
                calls.append(latent)
            samples = latent["samples"] if isinstance(latent, dict) else latent
            if getattr(samples, "is_nested", False):
                parts = list(samples.unbind())
                return type(
                    "NestedNoise",
                    (),
                    {"is_nested": True, "unbind": lambda self: list(parts)},
                )()
            return samples

    return FakeNoise()


def make_recording_guider(*, sigma_calls=None, callback_every_step=True, stage_shapes=None):
    """Echo guider that records full sigma schedules and stage latent geometry."""
    class Model:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

        def process_latent_out(self, x):
            return x

    class Guider:
        model_patcher = type("ModelPatcher", (), {"model": Model()})()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if sigma_calls is not None:
                sigma_calls.append([float(s) for s in sigmas])
            if stage_shapes is not None:
                video = next(
                    (part for part in latent_image.unbind() if getattr(part, "ndim", 0) == 5),
                    None,
                )
                stage_shapes.append(tuple(video.shape[-2:]) if video is not None else None)
            if callback is not None:
                count = len(sigmas) - 1
                if callback_every_step:
                    for i in range(count):
                        callback(i, latent_image, latent_image, count)
                elif count:
                    callback(0, latent_image, latent_image, count)
            return latent_image

    return Guider()


@pytest.fixture(scope="session", autouse=True)
def _comfy_stubs():
    install_comfy_stubs()
    return None
