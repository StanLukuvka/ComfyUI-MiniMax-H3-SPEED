"""CPU tests for per-callback Sigma Trace measurements (no H3 model required)."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
NODES_DIR = REPO_ROOT / "nodes"

# install stubs before any module import
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NODES_DIR))

import math

_comfy = ModuleType("comfy")
_samplers = ModuleType("comfy.samplers")
_utils = ModuleType("comfy.utils")
_model_mgmt = ModuleType("comfy.model_management")
_kdiff = ModuleType("comfy.k_diffusion")
_ksampling = ModuleType("comfy.k_diffusion.sampling")
_nested_tensor = ModuleType("comfy.nested_tensor")

class NestedTensor:
    is_nested = True
    def __init__(self, tensors):
        self._tensors = list(tensors)
    def unbind(self):
        return self._tensors

_nested_tensor.NestedTensor = NestedTensor
_samplers.sampler_object = lambda name: ("sampler", name)
_utils.PROGRESS_BAR_ENABLED = True

class _ProgressBar:
    def __init__(self, total, node_id=None):
        self.total = total
        self.node_id = node_id
    def update_absolute(self, value, total=None, preview=None):
        pass
    def update(self, value):
        pass

_utils.ProgressBar = _ProgressBar

def _pack_latents(latents):
    shapes, tensors = [], []
    for t in latents:
        shapes.append(list(t.shape))
        tensors.append(t.reshape(t.shape[0], 1, -1))
    return torch.cat(tensors, dim=-1), shapes

def _unpack_latents(combined, shapes):
    out, work = [], combined
    for shape in shapes:
        cut = math.prod(shape[1:])
        out.append(work[:, :, :cut].reshape([work.shape[0]] + shape[1:]))
        work = work[:, :, cut:]
    return out

_utils.pack_latents = _pack_latents
_utils.unpack_latents = _unpack_latents
_model_mgmt.intermediate_device = lambda: "cpu"

_comfy.samplers = _samplers
_comfy.utils = _utils
_comfy.model_management = _model_mgmt
_comfy.k_diffusion = _kdiff
_comfy.k_diffusion.sampling = _ksampling
_comfy.nested_tensor = _nested_tensor
sys.modules["comfy"] = _comfy
for _name, _mod in [
    ("samplers", _samplers), ("utils", _utils), ("model_management", _model_mgmt),
    ("k_diffusion", _kdiff), ("k_diffusion.sampling", _ksampling),
    ("nested_tensor", _nested_tensor),
]:
    sys.modules["comfy." + _name] = _mod


class Nested:
    is_nested = True
    def __init__(self, tensors):
        self.tensors = list(tensors)
    def unbind(self):
        return self.tensors


class Noise:
    seed = 73
    def __init__(self, samples):
        self.samples = samples
        self.calls = []
    def generate_noise(self, latent):
        self.calls.append(latent)
        return self.samples


class EulerGuider:
    """Match the real ComfyUI CFGGuider.sample call signature.

    ComfyUI calls ``callback(step, x0, x, total_steps)`` — positional, not kwargs.
    """

    def __init__(self):
        self.model_patcher = object()
        self.calls = []
        self.predictions = []

    def sample(self, noise, latent_image, sampler, sigmas,
               denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.calls.append((noise, latent_image, sampler, sigmas,
                           dict(denoise_mask=denoise_mask, disable_pbar=disable_pbar, seed=seed)))
        if hasattr(noise, "is_nested") and noise.is_nested:
            video_t, audio_t = noise.unbind()
        else:
            video_t, audio_t = noise, noise
        if hasattr(latent_image, "is_nested") and latent_image.is_nested:
            lat_v, lat_a = latent_image.unbind()
        else:
            lat_v, lat_a = latent_image, latent_image
        states = [video_t.clone(), audio_t.clone()]
        n_steps = len(sigmas) - 1
        for step in range(n_steps):
            # x0 prediction: scale by step fraction so we get non-trivial RMS
            denoised = Nested([
                states[0].tanh() * (step + 1) / n_steps,
                states[1].tanh() * (step + 1) / n_steps,
            ])
            self.predictions.append(denoised)
            if callback is not None:
                # Real ComfyUI positional: (step, x0, x, total_steps)
                callback(step, denoised, Nested(states), n_steps)
            # Euler step
            sigma = sigmas[step].item() if hasattr(sigmas[step], "item") else float(sigmas[step])
            sigma_next = sigmas[step + 1].item() if hasattr(sigmas[step + 1], "item") else float(sigmas[step + 1])
            d = [(s - p) / sigma for s, p in zip(states, denoised.unbind())]
            states = [s + dn * (sigma_next - sigma) for s, dn in zip(states, d)]
        self.result = Nested(states)
        return self.result


@pytest.fixture
def trace_env(monkeypatch):
    previews = []
    preview_module = ModuleType("latent_preview")

    def prepare_callback(patcher, steps, x0_output_dict=None):
        def callback(step, x0, x, total_steps):
            previews.append((step, x0, x, total_steps))
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0
        return callback

    preview_module.prepare_callback = prepare_callback
    monkeypatch.setitem(sys.modules, "latent_preview", preview_module)

    spec = importlib.util.spec_from_file_location("sampler_sigma_trace_node", NODES_DIR / "sampler_sigma_trace_node.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, previews=previews, preview_module=preview_module)


def test_trace_runs_one_native_pass_and_preserves_every_callback(trace_env):
    cls = trace_env.module.MiniMaxH3SigmaTrace
    assert cls.INPUT_TYPES() == {"required": {
        "noise": ("NOISE",), "guider": ("GUIDER",), "sigmas": ("SIGMAS",),
        "latent_image": ("LATENT",),
    }}
    assert cls.RETURN_TYPES == ("STRING", "LATENT")
    assert cls.RETURN_NAMES == ("trace_json", "diagnostic_latent")

    video = torch.linspace(-1, 1, 24 * 3 * 6 * 8).reshape(1, 24, 3, 6, 8)
    audio = torch.linspace(-0.5, 0.5, 32 * 9).reshape(1, 32, 9)
    latent_samples = Nested([torch.zeros_like(video), torch.zeros_like(audio)])
    mask = torch.ones(1, 1, 3, 6, 8)
    latent = {"samples": latent_samples, "noise_mask": mask, "metadata": "keep"}
    noise = Noise(Nested([video, audio]))
    guider = EulerGuider()
    sigmas = torch.tensor([1.0, 0.7, 0.25, 0.0])

    json_text, output = cls().trace(noise, guider, sigmas, latent)
    document = json.loads(json_text)
    records = document["records"]
    assert document["schema_version"] == 1
    assert document["sampler"] == "euler"
    assert document["measured_tensor"] == "denoised_x0_video"
    assert document["sigmas"] == sigmas.tolist()
    assert document["expected_steps"] == len(sigmas) - 1
    assert document["callback_count"] == len(records) == len(sigmas) - 1
    assert document["complete"] is True
    assert [entry["step_index"] for entry in records] == list(range(len(sigmas) - 1))
    assert [entry["sigma"] for entry in records] == sigmas[:-1].tolist()
    assert [entry["sigma_next"] for entry in records] == sigmas[1:].tolist()
    assert len(trace_env.previews) == len(records)

    for step, entry in enumerate(records):
        predicted_video = guider.predictions[step].unbind()[0]
        assert entry["status"] == "ok"
        assert entry["sigma_source"] == "schedule"
        assert entry["latent_shapes"]["x0"] == [list(video.shape), list(audio.shape)]
        assert entry["video_shape"] == list(video.shape)
        assert entry["signal"]["rms"] == pytest.approx(
            predicted_video.square().mean().sqrt().item(), rel=1e-5
        )
        assert set(entry["spatial_dct"]["bands"]) == {"low", "mid", "high"}
        assert entry["temporal_dct"]["available"] is True
        assert len(entry["temporal_dct"]["power"]) == video.shape[2]
        assert trace_env.previews[step][0] == step
        assert trace_env.previews[step][1] is guider.predictions[step]
        assert trace_env.previews[step][3] == len(records)

    assert len(noise.calls) == len(guider.calls) == 1
    _, _, _, _, kwargs = guider.calls[0]
    assert kwargs["disable_pbar"] is False
    assert kwargs["seed"] == noise.seed
    assert output is not latent
    assert output["samples"] is guider.result
    assert output["noise_mask"] is mask
    assert output["metadata"] == "keep"
    assert latent["samples"] is latent_samples
    assert not {"A", "beta", "noise_amplitude", "noise_decay_exponent", "calibration"} & document.keys()
