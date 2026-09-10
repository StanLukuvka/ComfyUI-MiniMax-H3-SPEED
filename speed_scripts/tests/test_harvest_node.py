"""Native-Euler Sigma Harvest node contracts."""

import importlib
import json

import torch

from conftest import make_nested


class Noise:
    seed = 42

    def generate_noise(self, latent):
        return latent["samples"] if isinstance(latent, dict) else latent


def _latent():
    video = torch.randn(1, 4, 8, 16, 16)
    audio = torch.zeros(1, 1, 2, 44)
    return {"samples": make_nested(video, audio), "metadata": "keep"}


def test_harvest_runs_one_native_euler_pass_and_emits_calibration():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    calls = []

    class Guider:
        model_patcher = type(
            "Patcher",
            (),
            {"model": type(
                "Model",
                (),
                {
                    "sigma_shift_video": 12.0,
                    "sigma_shift_audio": 3.0,
                    "process_latent_out": lambda self, x: x,
                },
            )()},
        )()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            calls.append(1)
            video = next(part for part in latent_image.unbind() if part.ndim == 5)
            state = video.float()
            total = len(sigmas) - 1
            for step in range(total):
                sigma = float(sigmas[step])
                denoised = state * .5
                derivative = (state - denoised) / sigma if sigma else (state - denoised)
                state = state + derivative * (float(sigmas[step + 1]) - sigma)
                if callback is not None:
                    callback(step, denoised, state, total)
            return state

    required = cls.INPUT_TYPES()["required"]
    assert {"noise", "guider", "sigmas", "latent_image"} <= set(required)
    assert cls.RETURN_TYPES == ("STRING", "LATENT")

    text, diagnostic = cls().harvest(
        Noise(),
        Guider(),
        torch.linspace(1.0, .025, 20),
        _latent(),
        delta=.01,
    )
    calibration = json.loads(text)

    assert len(calls) == 1
    assert calibration["schema_version"] == 2
    assert calibration["measurement_basis"] == "residual_x_minus_denoised"
    assert calibration["calibration_kind"] == "empirical_h3_residual_fit"
    assert calibration["delta"] == .01
    assert 0 < calibration["noise_amplitude"] < 1e6
    assert -5 <= calibration["noise_decay_exponent"] < 10
    assert {"r2", "health", "report"} <= set(calibration)
    assert isinstance(diagnostic, dict) and "samples" in diagnostic

    # Paste line mirrors the Automatic widget's 4-decimal A/beta precision;
    # delta stays at 3 decimals.
    paste_line = next(
        line for line in calibration["report"].splitlines()
        if line.startswith("Paste into SPEED Sampler:")
    )
    assert "noise_amplitude=" + format(calibration["noise_amplitude"], ".4f") in paste_line
    assert "noise_decay_exponent=" + format(calibration["noise_decay_exponent"], ".4f") in paste_line
    assert "Tolerance (Delta)=" + format(calibration["delta"], ".3f") in paste_line


def test_harvest_reports_no_captures_instead_of_inventing_fit():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig

    class SilentGuider:
        model_patcher = type(
            "Patcher",
            (),
            {"model": type(
                "Model",
                (),
                {
                    "sigma_shift_video": 12.0,
                    "sigma_shift_audio": 3.0,
                    "process_latent_out": lambda self, x: x,
                },
            )()},
        )()

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            return latent_image

    text, diagnostic = cls().harvest(
        Noise(),
        SilentGuider(),
        torch.linspace(1.0, .025, 20),
        _latent(),
    )
    assert json.loads(text)["error"] == "no_captures"
    assert isinstance(diagnostic, dict)
