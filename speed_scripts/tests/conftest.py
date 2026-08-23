"""pytest bootstrap for the reorganised layout.

The library package moved from ``minimax_h3_speed/`` to ``speed_scripts/`` and
the node modules moved from the repo root into ``nodes/``. Put the repo root
and the ``nodes`` directory on ``sys.path`` before collection so both the
``speed_scripts.*`` package imports and the flat node-module imports
(``sampler_node``, ``sampler_node_manual``, ``sampler_sigma_harvest_node``)
resolve.

Also hosts the ONE canonical comfy-stub installer (the superset of what every
test module used to hand-maintain in six drifting copies) plus shared fakes.
"""

import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../minimax_speed
NODES_DIR = REPO_ROOT / "nodes"

for _p in (str(REPO_ROOT), str(NODES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def install_comfy_stubs():
    """Install minimal comfy.* module stubs so node/runtime modules import
    without a live ComfyUI. Idempotent: re-registers fresh stubs every call
    (node modules imported after the first call keep their sys.modules view,
    which is fine — the stubs are stateless).
    """
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
            self._tensors = tensors
        def unbind(self):
            return self._tensors
    nested_tensor.NestedTensor = NestedTensor

    samplers.sampler_object = lambda name: ("sampler", name)
    utils.PROGRESS_BAR_ENABLED = True

    class _ProgressBar:
        """No-op stand-in for comfy.utils.ProgressBar in headless tests."""
        def __init__(self, total, node_id=None):
            self.total = total
            self.node_id = node_id
        def update_absolute(self, value, total=None, preview=None):
            pass
        def update(self, value):
            pass
    utils.ProgressBar = _ProgressBar

    def pack_latents(latents):
        shapes, tensors = [], []
        for t in latents:
            shapes.append(list(t.shape))
            tensors.append(t.reshape(t.shape[0], 1, -1))
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
            x = x + d * (sigmas[i + 1] - sigmas[i])
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
    for name, mod in [("samplers", samplers), ("utils", utils),
                      ("model_management", model_mgmt),
                      ("k_diffusion", kdiff), ("k_diffusion.sampling", ksampling),
                      ("nested_tensor", nested_tensor)]:
        sys.modules["comfy." + name] = mod


def make_fake_guider(calls_list=None):
    """FakeGuider with an H3-shaped model_patcher.

    sample() echoes the latent and records len(sigmas) into calls_list.
    """
    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if calls_list is not None:
                calls_list.append(len(sigmas))
            if callback is not None:
                callback(0, latent_image, latent_image, len(sigmas))
            return latent_image

    return FakeGuider()


def make_fake_noise(seed_value=42, captured=None):
    """FakeNoise that regenerates the (video, audio) streams unchanged."""
    class FakeNoise:
        seed = seed_value
        def generate_noise(self, latent):
            samples = latent.get("samples")
            if getattr(samples, "is_nested", False):
                vids = [s for s in samples.unbind() if s.ndim == 5]
                auds = [s for s in samples.unbind() if s.ndim != 5]
                nt = type("NT", (), {"is_nested": True, "unbind": lambda self: vids + auds})()
                if captured is not None:
                    captured.append(nt)
                return nt
            return samples

    return FakeNoise()


def make_nested(video, audio):
    return type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()


@pytest.fixture(scope="session", autouse=True)
def _comfy_stubs(doctest_namespace=None):
    """Install comfy stubs before any test so modules importing comfy work."""
    install_comfy_stubs()
    return None
