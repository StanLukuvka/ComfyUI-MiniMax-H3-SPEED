"""Shared pytest bootstrap and ComfyUI fakes for the SPEED test suite."""

import sys
from pathlib import Path
from types import ModuleType

import torch

REPO_ROOT = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "nodes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class NestedTensor:
    """Stand-in for ComfyUI's NestedTensor (the H3 video/audio latent bundle)."""

    is_nested = True

    def __init__(self, tensors):
        self._tensors = list(tensors)

    def unbind(self):
        return list(self._tensors)


def install_comfy_stubs():
    """Install the minimal comfy.* surface the nodes and runtime import."""
    utils = ModuleType("comfy.utils")
    utils.PROGRESS_BAR_ENABLED = True
    utils.ProgressBar = type("ProgressBar", (), {
        "__init__": lambda self, total: None,
        "update_absolute": lambda self, value, total=None, preview=None: None,
    })
    nested_tensor = ModuleType("comfy.nested_tensor")
    nested_tensor.NestedTensor = NestedTensor
    sample = ModuleType("comfy.sample")
    # The real one widens generic empty latents to the model's channels; the fakes already match.
    sample.fix_empty_latent_channels = lambda model, latent_image, *ratios: latent_image

    model_sampling = ModuleType("comfy.model_sampling")

    def _sampling_class(name, bases=()):
        cls = type(name, bases, {"__module__": "comfy.model_sampling"})
        setattr(model_sampling, name, cls)
        return cls

    const = _sampling_class("CONST")
    _sampling_class("ModelSamplingDiscrete")
    _sampling_class("EPS")
    _sampling_class("IMG_TO_IMG_FLOW", (const,))
    _sampling_class("IMG_TO_IMG_VELOCITY", (const,))
    _sampling_class("ModelSamplingAV", (_sampling_class("ModelSamplingDiscreteFlow"),))

    comfy = ModuleType("comfy")
    for mod in (utils, nested_tensor, sample, model_sampling):
        name = mod.__name__.split(".")[1]
        setattr(comfy, name, mod)
        sys.modules[mod.__name__] = mod
    sys.modules["comfy"] = comfy


# Installed at import time so test modules can import node and runtime code directly.
install_comfy_stubs()

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])


def make_latent(*, h=8, w=8, t=2):
    """Empty H3 LATENT: nested [1,1,T,H,W] video and [1,1,2,44] audio."""
    return {"samples": NestedTensor([torch.zeros(1, 1, t, h, w), torch.zeros(1, 1, 2, 44)])}


def image_latent(shape):
    return {"samples": torch.zeros(shape)}


class EchoNoise:
    """NOISE that returns the latent itself (all zeros for an empty latent)."""

    seed = 42

    def generate_noise(self, latent):
        samples = latent["samples"]
        return NestedTensor(samples.unbind()) if getattr(samples, "is_nested", False) else samples


class SeededNoise:
    """NOISE drawing seeded Gaussian noise for plain or nested latents."""

    def __init__(self, seed=42):
        self.seed = seed

    def generate_noise(self, latent):
        samples = latent["samples"]
        generator = torch.Generator().manual_seed(self.seed)
        if getattr(samples, "is_nested", False):
            return NestedTensor(torch.randn(p.shape, generator=generator) for p in samples.unbind())
        return torch.randn(samples.shape, generator=generator)


class H3Model:
    """MiniMax-H3 model stand-in: the sigma shifts and an identity latent format."""

    sigma_shift_video = 12.0
    sigma_shift_audio = 3.0

    def __init__(self):
        self.latents_out = []

    def process_latent_out(self, x):
        self.latents_out.append(x)
        return x


class Patcher:
    """ModelPatcher stand-in; ``get_model_object`` serves ``objects`` as ComfyUI does."""

    def __init__(self, model, **objects):
        self.model = model
        self.objects = objects

    def get_model_object(self, name):
        return self.objects.get(name)


class EchoGuider:
    """H3 GUIDER whose every stage returns its input noise.

    Video output is the stage noise plus ``video_offset``; audio output is the
    stage noise, or ``audio_value`` everywhere when set. Each ``sample`` call
    records the sampler, seed, sigma schedule, input noise and latent size,
    and fires the callback once per step. ``fail_at_call`` makes that call raise.
    """

    def __init__(self, video_offset=0.0, audio_value=None, fail_at_call=None):
        self.model_patcher = Patcher(H3Model())
        self.video_offset = video_offset
        self.audio_value = audio_value
        self.fail_at_call = fail_at_call
        self.samplers, self.seeds, self.sigma_calls, self.noise, self.latent_sizes = [], [], [], [], []
        # Real guiders carry the conditioning here; I2V tests attach keyframes and refs.
        self.original_conds = None

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        if len(self.samplers) + 1 == self.fail_at_call:
            self.samplers.append(sampler)
            raise RuntimeError("stage exploded")
        video, audio = noise.unbind()
        self.samplers.append(sampler)
        self.seeds.append(kwargs.get("seed"))
        self.sigma_calls.append([float(s) for s in sigmas])
        self.noise.append((video.clone(), audio.clone()))
        self.latent_sizes.append(tuple(latent_image.unbind()[0].shape[-2:]))
        if self.audio_value is not None:
            audio = torch.full_like(audio, self.audio_value)
        out = NestedTensor([video + self.video_offset, audio])
        count = len(sigmas) - 1
        if callback is not None:
            for i in range(count):
                callback(i, out, out, count)
        return out


class StubSampler:
    """ComfyUI SAMPLER stand-in (as KSamplerSelect returns), named like ``sample_<name>``."""

    def __init__(self, name="euler"):
        def sampler_function(*args, **kwargs):
            raise AssertionError("StubSampler is not meant to be run")

        sampler_function.__name__ = f"sample_{name}"
        self.sampler_function = sampler_function


class AffineModel:
    """Flow image model whose latent format has a mean/std like Wan 2.1's (used by Krea 2)."""

    mean, std = 0.7, 2.0

    def process_latent_in(self, x):
        return (x - self.mean) / self.std

    def process_latent_out(self, x):
        return x * self.std + self.mean


def model_sampling(*base_names):
    """ModelSampling object shaped like ComfyUI's ``class ModelSampling(s, c)`` in model_base.py."""
    import comfy.model_sampling as ms

    names = base_names or ("ModelSamplingDiscrete", "CONST")
    return type("ModelSampling", tuple(getattr(ms, name) for name in names), {})()


def flow_patcher(sampling=None, latent_dimensions=2):
    """Patcher for a flow image model with the given sampling and latent-format rank."""
    latent_format = type("LatentFormat", (), {"latent_dimensions": latent_dimensions})()
    return Patcher(
        AffineModel(),
        model_sampling=model_sampling() if sampling is None else sampling,
        latent_format=latent_format,
    )


class OracleFlowGuider:
    """Mimics ComfyUI's CFGGuider.sample for a CONST flow model with an ideal denoiser.

    The zero latent is not shifted, the start state is ``sigma0 * noise``, the
    denoiser always predicts ``clean`` (low-passed to the stage size), and the
    result goes through ``inverse_noise_scaling`` and ``process_latent_out``.
    """

    def __init__(self, clean, latent_dimensions=2):
        self.model_patcher = flow_patcher(latent_dimensions=latent_dimensions)
        self.clean = clean
        self.starts, self.ends = [], []
        self.original_conds = None

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        from speed_scripts.spectral import lowpass_dct

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


def aligned(sigma, ratio):
    """SPEED's boundary alignment: ``kappa = r / (1 + (r - 1) q)``; returns ``(kappa, q * kappa)``."""
    kappa = ratio / (1.0 + (ratio - 1.0) * sigma)
    return kappa, sigma * kappa


def expected_stage_sigmas(sigmas, boundaries, scales):
    """Each stage's sigma schedule: global boundaries, each re-entry aligned by its scale ratio."""
    working = [float(s) for s in sigmas]
    starts, ends = [0, *boundaries], [*boundaries, len(working) - 1]
    schedules = []
    for stage, (start, end) in enumerate(zip(starts, ends)):
        schedules.append(working[start:end + 1])
        if stage < len(boundaries):
            _, working[end] = aligned(working[end], scales[stage + 1] / scales[stage])
    return schedules
