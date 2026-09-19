"""Sampler-aware native Sigma Harvest node contracts."""

import importlib
import inspect
import json

import pytest
import torch

from conftest import make_nested
from speed_scripts.sampler_support import SUPPORTED_SPEED_SAMPLERS


class Noise:
    seed = 42

    def generate_noise(self, latent):
        return latent["samples"] if isinstance(latent, dict) else latent


def _latent():
    video = torch.randn(1, 4, 8, 16, 16)
    audio = torch.zeros(1, 1, 2, 44)
    return {"samples": make_nested(video, audio), "metadata": "keep"}


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

    def __init__(self):
        self.samplers = []

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
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


def _harvest(cls, guider, sampler_name="euler", **kwargs):
    return cls().harvest(
        Noise(),
        guider,
        torch.linspace(1.0, .025, 20),
        _latent(),
        sampler_name=sampler_name,
        **kwargs,
    )


def test_input_contract_uses_shared_sampler_dropdown_and_signature():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    required = cls.INPUT_TYPES()["required"]
    optional = cls.INPUT_TYPES()["optional"]

    assert tuple(required["sampler_name"][0]) == SUPPORTED_SPEED_SAMPLERS
    assert required["sampler_name"][1]["default"] == "euler"
    assert list(required) == ["noise", "guider", "sigmas", "latent_image", "sampler_name"]
    assert "Tolerance (Delta)" in optional
    assert list(inspect.signature(cls.harvest).parameters)[5] == "sampler_name"
    assert inspect.signature(cls.harvest).parameters["sampler_name"].default == "euler"


@pytest.mark.parametrize("sampler_name", SUPPORTED_SPEED_SAMPLERS)
def test_harvest_uses_selected_native_sampler_and_emits_identity(monkeypatch, sampler_name):
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    native_calls = []

    def native_sampler_object(name):
        native_calls.append(name)
        return ("native-sampler", name)

    monkeypatch.setattr(
        importlib.import_module("comfy.samplers"),
        "sampler_object",
        native_sampler_object,
    )
    guider = Guider()
    text, diagnostic = _harvest(cls, guider, sampler_name, delta=.01)
    calibration = json.loads(text)

    assert native_calls == [sampler_name]
    assert guider.samplers == [("native-sampler", sampler_name)]
    assert calibration["sampler_name"] == sampler_name
    assert calibration["schema_version"] == 2
    assert calibration["measurement_basis"] == "residual_x_minus_denoised"
    assert calibration["calibration_kind"] == "empirical_h3_residual_fit"
    assert calibration["delta"] == .01
    assert 0 < calibration["noise_amplitude"] < 1e6
    assert -5 <= calibration["noise_decay_exponent"] < 10
    assert {"r2", "health", "report"} <= set(calibration)
    assert isinstance(diagnostic, dict) and "samples" in diagnostic

    report_lines = calibration["report"].splitlines()
    assert sampler_name in report_lines[0]
    if calibration["noise_decay_exponent"] > 0 and calibration["health"] != "invalid":
        paste_line = next(
            line for line in report_lines if line.startswith("Paste into SPEED Sampler:")
        )
        assert f"sampler_name={sampler_name}" in paste_line
        assert "noise_amplitude=" + format(calibration["noise_amplitude"], ".4f") in paste_line
        assert "noise_decay_exponent=" + format(calibration["noise_decay_exponent"], ".4f") in paste_line
        assert "Tolerance (Delta)=" + format(calibration["delta"], ".3f") in paste_line
    else:
        assert "Do not paste this calibration into Automatic" in calibration["report"]


def test_harvest_reduces_each_residual_during_callback(monkeypatch):
    module = importlib.import_module("sampler_sigma_harvest_node")
    cls = module.MiniMaxH3HarvestToConfig
    original = module.radial_dct_power
    reduced = []

    def recording_radial_dct_power(residual):
        reduced.append(tuple(residual.shape))
        return original(residual)

    monkeypatch.setattr(module, "radial_dct_power", recording_radial_dct_power)

    class StreamingGuider(Guider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            self.samplers.append(sampler)
            video = next(part for part in latent_image.unbind() if part.ndim == 5)
            state = video.float()
            total = len(sigmas) - 1
            for step in range(total):
                sigma = float(sigmas[step])
                denoised = state * .5
                derivative = (state - denoised) / sigma if sigma else (state - denoised)
                state = state + derivative * (float(sigmas[step + 1]) - sigma)
                callback(step, denoised, state, total)
                assert len(reduced) == step + 1
            return state

    text, _ = _harvest(cls, StreamingGuider(), "euler")
    assert "error" not in json.loads(text)
    assert len(reduced) == 19


def test_unusable_harvest_fit_is_not_reported_as_paste_ready(monkeypatch):
    module = importlib.import_module("sampler_sigma_harvest_node")
    cls = module.MiniMaxH3HarvestToConfig

    monkeypatch.setattr(
        module,
        "fit_power_law",
        lambda *args: {"A": 1.0, "beta": -0.5, "r_squared": 0.2, "n_bins": 8},
    )
    text, _ = _harvest(cls, Guider(), "euler")
    calibration = json.loads(text)

    assert calibration["health"] == "suspect"
    assert "Do not paste this calibration into Automatic" in calibration["report"]
    assert "Paste into SPEED Sampler:" not in calibration["report"]


def test_res_harvest_uses_native_sampler_stub_not_speed_adapter(monkeypatch):
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    native_calls = []

    def native_sampler_object(name):
        native_calls.append(name)
        return ("native-res", name)

    monkeypatch.setattr(
        importlib.import_module("comfy.samplers"),
        "sampler_object",
        native_sampler_object,
    )
    guider = Guider()
    _harvest(cls, guider, "res_multistep")

    assert native_calls == ["res_multistep"]
    assert guider.samplers == [("native-res", "res_multistep")]


def test_harvest_reports_no_captures_instead_of_inventing_fit():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig

    class SilentGuider(Guider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            self.samplers.append(sampler)
            return latent_image

    text, diagnostic = _harvest(cls, SilentGuider(), "heun")
    error = json.loads(text)
    assert error["error"] == "no_captures"
    assert error["sampler_name"] == "heun"
    assert isinstance(diagnostic, dict)


@pytest.mark.parametrize("error_kind", ["harvest_failed", "fit_failed"])
def test_error_json_escapes_exception_text(monkeypatch, error_kind):
    module = importlib.import_module("sampler_sigma_harvest_node")
    cls = module.MiniMaxH3HarvestToConfig
    message = 'bad "quote"\nnext line\\slash'

    if error_kind == "harvest_failed":
        class FailingGuider(Guider):
            def sample(self, *args, **kwargs):
                raise RuntimeError(message)

        text, _ = _harvest(cls, FailingGuider(), "dpm_2")
    else:
        monkeypatch.setattr(module, "fit_power_law", lambda *args: (_ for _ in ()).throw(ValueError(message)))
        text, _ = _harvest(cls, Guider(), "dpm_2")

    error = json.loads(text)
    assert error["error"] == error_kind
    assert message in error["message"]
    assert error["sampler_name"] == "dpm_2"
