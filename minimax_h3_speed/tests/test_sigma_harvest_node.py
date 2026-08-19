"""Regression test for the MiniMaxH3SigmaHarvest node's run() entry point.

The H3 guider's comfy.samplers sample() calls `noise.unbind()` directly,
so the node MUST pass a pre-generated NestedTensor (from
`noise.generate_noise(latent)`), not the raw Noise_RandomNoise generator.

This test models that contract in the stubs: if run() ever reverts to
passing the raw generator, the stub guider raises the exact
`AttributeError: 'Noise_RandomNoise' object has no attribute 'unbind'`
seen on Kaggle (ComfyUI v0.33.1).
"""

import importlib
import sys
from types import ModuleType

import pytest
import torch


def _install_comfy_stubs():
    comfy = ModuleType("comfy")
    samplers = ModuleType("comfy.samplers")
    utils = ModuleType("comfy.utils")
    model_mgmt = ModuleType("comfy.model_management")
    nested_tensor = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True

        def __init__(self, tensors):
            self._t = tensors

        def unbind(self):
            return self._t

    nested_tensor.NestedTensor = NestedTensor

    # Functions/attrs the node touches via comfy.samplers / comfy.utils.
    def sampler_object(name):
        return ("sampler", name)

    samplers.sampler_object = sampler_object
    utils.PROGRESS_BAR_ENABLED = True
    model_mgmt.intermediate_device = lambda: "cpu"

    # H3 guider sample(): calls noise.unbind() directly (samplers.py:1283).
    class _H3Guider:
        def sample(self, noise, latent_image, sampler, sigmas, callback=None,
                   disable_pbar=True, seed=None):
            noise.unbind()
            if callback is not None:
                # Model a real denoise step: x0/x are NestedTensors with a
                # non-zero residual so the spectrum has power above omega_min.
                nt = sys.modules["comfy.nested_tensor"].NestedTensor
                x0 = nt([torch.randn(1, 4, 8, 64, 64), torch.randn(1, 4, 2, 8)])
                x = nt([torch.randn(1, 4, 8, 64, 64), torch.randn(1, 4, 2, 8)])
                callback(0, x0, x, len(sigmas))
            return latent_image

    comfy.samplers = samplers
    comfy.utils = utils
    comfy.model_management = model_mgmt
    comfy.nested_tensor = nested_tensor

    sys.modules["comfy"] = comfy
    sys.modules["comfy.samplers"] = samplers
    sys.modules["comfy.utils"] = utils
    sys.modules["comfy.model_management"] = model_mgmt
    sys.modules["comfy.nested_tensor"] = nested_tensor
    return nested_tensor, samplers, _H3Guider


_nested_tensor, _samplers, _H3Guider = _install_comfy_stubs()


class _FakeNoise:
    """Emulates comfy_extras.nodes_custom_sampler.Noise_RandomNoise."""

    def __init__(self, seed=42):
        self.seed = seed
        self.generate_calls = 0

    def generate_noise(self, latent):
        self.generate_calls += 1
        samples = latent.get("samples")
        video = torch.randn(1, 4, 8, 64, 64)
        audio = torch.randn(1, 4, 2, 8)
        return sys.modules["comfy.nested_tensor"].NestedTensor([video, audio])


class _FakeNested:
    is_nested = True

    def unbind(self):
        return [torch.zeros(1, 4, 8, 64, 64), torch.zeros(1, 4, 2, 8)]


def test_sigma_harvest_callable_spot():
    """run() must call noise.generate_noise() and pass a NestedTensor on."""
    mod = importlib.import_module("sigma_harvest_node")

    noise = _FakeNoise()
    latent_image = {"samples": _FakeNested()}
    # 20 sigmas -> 20 captures. 64x64 gives enough frequency bins for the fitter.
    sigmas = torch.linspace(1.0, 0.0, 21)

    out = mod.MiniMaxH3SigmaHarvest().run(
        noise,
        _H3Guider(),
        sigmas,
        latent_image,
        capture_every=1,
        fit_mode="first",
        omega_min=0.5,
        delta=0.01,
    )
    # generate_noise must have been called exactly once.
    assert noise.generate_calls == 1
    # Node returns a JSON string with a fitted power law.
    assert isinstance(out, tuple) and len(out) == 1
    payload = out[0]
    assert isinstance(payload, str)
    assert "overall_fit_A" in payload
    assert "overall_fit_beta" in payload
