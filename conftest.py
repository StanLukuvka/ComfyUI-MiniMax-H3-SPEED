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
    utils = ModuleType("comfy.utils")
    model_mgmt = ModuleType("comfy.model_management")
    nested_tensor = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True

        def __init__(self, tensors):
            self._tensors = list(tensors)

        def unbind(self):
            return list(self._tensors)

    nested_tensor.NestedTensor = NestedTensor
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
    sample = ModuleType("comfy.sample")
    # The real one widens generic empty latents to the model's channels; the fakes already match.
    sample.fix_empty_latent_channels = lambda model, latent_image, *ratios: latent_image
    model_sampling = ModuleType("comfy.model_sampling")

    def _sampling_class(name, bases=()):
        return type(name, bases, {"__module__": "comfy.model_sampling"})

    CONST = _sampling_class("CONST")
    model_sampling.ModelSamplingDiscrete = _sampling_class("ModelSamplingDiscrete")
    model_sampling.EPS = _sampling_class("EPS")
    model_sampling.CONST = CONST
    model_sampling.IMG_TO_IMG_FLOW = _sampling_class("IMG_TO_IMG_FLOW", (CONST,))
    model_sampling.IMG_TO_IMG_VELOCITY = _sampling_class("IMG_TO_IMG_VELOCITY", (CONST,))

    comfy.utils = utils
    comfy.model_management = model_mgmt
    comfy.nested_tensor = nested_tensor
    comfy.sample = sample
    comfy.model_sampling = model_sampling
    sys.modules["comfy"] = comfy
    for name, mod in (
        ("utils", utils),
        ("model_management", model_mgmt),
        ("nested_tensor", nested_tensor),
        ("sample", sample),
        ("model_sampling", model_sampling),
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


class RecordingEchoGuider:
    """Echo guider that records what every stage call received.

    The public output is the stage's noise video plus a fixed offset, so a
    finished run carries non-trivial signal. Records the sampler object, the
    sigma schedule, the noise seed, and the noise geometry of each ``sample``
    call.
    """

    class Model:
        sigma_shift_video = 12.0
        sigma_shift_audio = 3.0

        def process_latent_out(self, x):
            return x

    def __init__(self, video_offset=0.0):
        self.model_patcher = type("P", (), {"model": self.Model()})()
        self.video_offset = video_offset
        self.samplers = []
        self.seeds = []
        self.sigma_calls = []
        self.noise_shapes = []
        # Real ComfyUI guiders carry the conditioning dict here; I2V tests
        # attach a shaped fake (minimax_keyframes / minimax_refs).
        self.original_conds = None

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        self.seeds.append(kwargs.get("seed"))
        self.sigma_calls.append([float(s) for s in sigmas])
        pub_video, pub_audio = list(noise.unbind())
        self.noise_shapes.append(tuple(pub_video.shape))
        out = make_nested(pub_video + self.video_offset, pub_audio)
        count = len(sigmas) - 1
        if callback is not None:
            for i in range(count):
                callback(i, out, out, count)
        return out


class StubSampler:
    """Minimal stand-in for a ComfyUI SAMPLER (as KSamplerSelect returns).

    Echo guiders never call it; it is an identity token that guiders record.
    ``sampler_function.__name__`` follows ComfyUI's ``sample_<name>`` form.
    """

    def __init__(self, name="euler"):
        def sampler_function(*args, **kwargs):
            raise AssertionError("StubSampler is not meant to be run")

        sampler_function.__name__ = f"sample_{name}"
        self.name = name
        self.sampler_function = sampler_function

    def __repr__(self):
        return f"StubSampler({self.name!r})"


@pytest.fixture(scope="session", autouse=True)
def _comfy_stubs():
    install_comfy_stubs()
    return None
