"""Shared pytest bootstrap and ComfyUI fakes for the SPEED test suite."""

import sys
from pathlib import Path
from types import ModuleType

import torch

REPO_ROOT = Path(__file__).resolve().parent
NODES_DIR = REPO_ROOT / "nodes"
for _p in (str(REPO_ROOT), str(NODES_DIR)):
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
    comfy = ModuleType("comfy")
    utils = ModuleType("comfy.utils")
    nested_tensor = ModuleType("comfy.nested_tensor")

    nested_tensor.NestedTensor = NestedTensor
    utils.PROGRESS_BAR_ENABLED = True

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total

        def update_absolute(self, value, total=None, preview=None):
            pass

    utils.ProgressBar = ProgressBar

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
    flow = _sampling_class("ModelSamplingDiscreteFlow")
    model_sampling.ModelSamplingAV = _sampling_class("ModelSamplingAV", (flow,))

    modules = {
        "utils": utils,
        "nested_tensor": nested_tensor,
        "sample": sample,
        "model_sampling": model_sampling,
    }
    for name, mod in modules.items():
        setattr(comfy, name, mod)
        sys.modules[f"comfy.{name}"] = mod
    sys.modules["comfy"] = comfy


# Installed at import time so test modules can import node and runtime code directly.
install_comfy_stubs()


# ---------------------------------------------------------------------------
# Latents and noise
# ---------------------------------------------------------------------------

SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])



def make_latent(*, h=8, w=8, t=2, channels=1):
    """Empty H3 LATENT: nested [1,C,T,H,W] video and [1,1,2,44] audio."""
    return {"samples": NestedTensor([torch.zeros(1, channels, t, h, w), torch.zeros(1, 1, 2, 44)])}


def image_latent(shape):
    """Empty plain image LATENT."""
    return {"samples": torch.zeros(shape)}


class EchoNoise:
    """NOISE that returns the latent itself (all zeros for an empty latent)."""

    seed = 42

    def generate_noise(self, latent):
        samples = latent["samples"]
        if getattr(samples, "is_nested", False):
            return NestedTensor(samples.unbind())
        return samples


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


# ---------------------------------------------------------------------------
# Models, patchers, guiders, samplers
# ---------------------------------------------------------------------------

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

    The video output is the stage noise plus ``video_offset``; the audio output
    is the stage noise, or ``audio_value`` everywhere when set. Each ``sample``
    call records the sampler, seed, sigma schedule, input noise, and
    conditioning-latent size, and fires the callback once per step.
    """

    def __init__(self, video_offset=0.0, audio_value=None):
        self.model_patcher = Patcher(H3Model())
        self.video_offset = video_offset
        self.audio_value = audio_value
        self.samplers = []
        self.seeds = []
        self.sigma_calls = []
        self.noise = []
        self.latent_sizes = []
        # Real guiders carry the conditioning here; I2V tests attach keyframes and refs.
        self.original_conds = None

    @property
    def noise_shapes(self):
        return [tuple(video.shape) for video, _ in self.noise]

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
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
    """ComfyUI SAMPLER stand-in (as KSamplerSelect returns); guiders only record it.

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


# ---------------------------------------------------------------------------
# Flow image models
# ---------------------------------------------------------------------------

class AffineModel:
    """Flow image model whose latent format has a mean/std like Wan 2.1's (used by Krea 2)."""

    mean, std = 0.7, 2.0

    def process_latent_in(self, x):
        return (x - self.mean) / self.std

    def process_latent_out(self, x):
        return x * self.std + self.mean


def model_sampling(*bases):
    """ModelSampling object shaped like ComfyUI's ``class ModelSampling(s, c)`` in model_base.py."""
    import comfy.model_sampling

    bases = bases or (comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.CONST)
    return type("ModelSampling", bases, {})()


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

    def __init__(self, clean, latent_dimensions=2, sampling=None):
        self.model_patcher = flow_patcher(sampling, latent_dimensions)
        self.clean = clean
        self.starts = []
        self.ends = []
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


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------

def expected_stage_sigmas(sigmas, boundaries, scales):
    """Each stage's sigma schedule: global boundaries, each re-entry aligned by its scale ratio."""
    from speed_scripts.flow import aligned_sigma

    working = [float(s) for s in sigmas]
    starts = [0, *boundaries]
    ends = [*boundaries, len(working) - 1]
    schedules = []
    for stage, (start, end) in enumerate(zip(starts, ends)):
        schedules.append(list(working[start:end + 1]))
        if stage < len(boundaries):
            _, working[end] = aligned_sigma(working[end], scales[stage + 1] / scales[stage])
    return schedules


def capture_node_config(monkeypatch, node_cls, **inputs):
    """Call a SPEED node with the pipeline stubbed out; return the SpeedConfig it built."""
    import importlib

    module = importlib.import_module(node_cls.__module__)
    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kwargs):
        captured["config"] = config
        return latent_image, latent_image

    monkeypatch.setattr(module, "run_speed_pipeline", fake_pipeline)
    inputs.setdefault("sampler", StubSampler())
    node_cls().sample(
        noise=EchoNoise(),
        guider=EchoGuider(),
        sigmas=torch.linspace(1.0, 0.0, 11),
        latent_image=make_latent(),
        **inputs,
    )
    return captured["config"]


def attach_i2v(guider):
    """Give ``guider`` two random 8x8 I2V keyframes and two refs; return ``(keyframes, refs)``."""
    keyframes = [{"latent": torch.rand(1, 1, 2, 8, 8)} for _ in range(2)]
    refs = [{"latent": torch.rand(1, 1, 2, 8, 8)} for _ in range(2)]
    guider.original_conds = {
        "positive": [{"minimax_keyframes": keyframes, "minimax_refs": refs}],
        "negative": [],
    }
    return keyframes, refs
