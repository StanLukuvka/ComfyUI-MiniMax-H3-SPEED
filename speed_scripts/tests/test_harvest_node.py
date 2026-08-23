"""Tests for the MiniMaxH3HarvestToConfig node — native-Euler sigma harvester."""

import importlib
import json
import math

import pytest
import torch

# Canonical comfy stubs (superset: progress bar + pack/unpack + euler) live in
# conftest now.
from conftest import install_comfy_stubs as _install_comfy_stubs


_install_comfy_stubs()


def test_harvest_node_inputs_match_native_sampler():
    """The harvest node must take the SAME inputs as a native sampler,
    not a dead harvest_json STRING input."""
    mod = importlib.import_module("sampler_sigma_harvest_node")
    cls = mod.MiniMaxH3HarvestToConfig
    required = cls.INPUT_TYPES()["required"]
    for key in ("noise", "guider", "sigmas", "latent_image"):
        assert key in required, f"harvest node missing native input: {key}"
    assert "harvest_json" not in required, "harvest node must NOT take a harvest_json STRING input"
    assert cls.RETURN_TYPES[0] == "STRING"
    assert cls.RETURN_TYPES == ("STRING", "LATENT")
    assert cls.RETURN_NAMES == ("calibration", "diagnostic_latent")


def test_harvest_node_runs_native_euler_and_emits_json():
    """Run the harvester with a synthetic guider and verify it captures
    residuals and emits a valid harvest_json with fitted A/beta."""
    mod = importlib.import_module("sampler_sigma_harvest_node")
    cls = mod.MiniMaxH3HarvestToConfig

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            # Unpack the nested latent to a tensor (as ComfyUI's guider does).
            # Signature must match real ComfyUI: callback(step, denoised, x, total_steps)
            if hasattr(latent_image, "is_nested") and latent_image.is_nested:
                vid = [s for s in latent_image.unbind() if s.ndim == 5][0]
            else:
                vid = latent_image
            x = vid.float()
            total = len(sigmas) - 1
            for i in range(total):
                sigma = sigmas[i]
                denoised = x * 0.5  # simple model: half the signal is denoised
                d = (x - denoised) / float(sigma) if float(sigma) != 0 else (x - denoised)
                x = x + d * (sigmas[i + 1] - sigma)
                if callback is not None:
                    # Real ComfyUI positional signature: (step, denoised, x, total_steps)
                    try:
                        callback(i, denoised, x, total)
                    except TypeError:
                        # Fallback for compat shim if test fake still uses dict style
                        callback({"x": x, "i": i, "sigma": float(sigma), "denoised": denoised})
            return x

    class FakeNoise:
        seed = 42
        def generate_noise(self, latent):
            return latent

    video = torch.randn(1, 4, 8, 16, 16)  # [B, C, T, H, W]
    audio = torch.zeros(1, 1, 2, 44)

    class FakeNested:
        is_nested = True
        def unbind(self):
            return [video, audio]

    nested = FakeNested()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.025, 20)

    node = cls()
    calibration_json, out_latent = node.harvest(
        FakeNoise(), FakeGuider(), sigmas, latent, delta=0.01
    )
    parsed = json.loads(calibration_json)
    # Plug-and-play calibration: flat JSON with noise_amplitude/beta
    assert "noise_amplitude" in parsed
    assert "noise_decay_exponent" in parsed
    assert "delta" in parsed
    assert "r2" in parsed
    assert "health" in parsed
    assert "report" in parsed
    # The fit should be a float in a reasonable range
    assert 0.0 < float(parsed["noise_amplitude"]) < 1e6
    assert -5.0 <= float(parsed["noise_decay_exponent"]) < 10.0
    # diagnostic_latent must be a LATENT dict, not bare tensor
    assert isinstance(out_latent, dict) and "samples" in out_latent


def test_harvest_node_no_captures_returns_error_json():
    """If the sampler callback never fires, emit explicit error JSON, not fake fit."""
    mod = importlib.import_module("sampler_sigma_harvest_node")
    cls = mod.MiniMaxH3HarvestToConfig

    class FakeGuiderNoCallback:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            return latent_image  # never calls callback

    class FakeNoise:
        seed = 42
        def generate_noise(self, latent):
            return latent

    video = torch.randn(1, 4, 8, 16, 16)
    audio = torch.zeros(1, 1, 2, 44)

    class FakeNested:
        is_nested = True
        def unbind(self):
            return [video, audio]

    nested = FakeNested()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.025, 20)

    node = cls()
    calibration_json, out_latent = node.harvest(
        FakeNoise(), FakeGuiderNoCallback(), sigmas, latent
    )
    parsed = json.loads(calibration_json)
    assert "error" in parsed
    assert parsed["error"] == "no_captures"
    assert isinstance(out_latent, dict)
