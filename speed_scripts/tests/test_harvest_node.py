"""Sampler-aware native Sigma Harvest node contracts."""

import importlib
import json

import pytest
import torch

from conftest import StubSampler, make_nested


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


def _harvest(cls, guider, sampler=None, **kwargs):
    return cls().harvest(
        noise=Noise(),
        guider=guider,
        sampler=sampler if sampler is not None else StubSampler(),
        sigmas=torch.linspace(1.0, .025, 20),
        latent_image=_latent(),
        **kwargs,
    )


def test_tolerance_delta_defaults_to_the_baked_value():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    assert cls.INPUT_TYPES()["optional"]["Tolerance (Delta)"][1]["default"] == 0.005


class _ClassNamedSampler:
    """A SAMPLER without ``sampler_function``; labelled by its class name."""


class _FunctionSuffixSampler:
    """A SAMPLER whose function is named ``<name>_function`` (e.g. dpm_fast)."""

    def __init__(self, name):
        def sampler_function(*args, **kwargs):
            raise AssertionError("not meant to be run")

        sampler_function.__name__ = f"{name}_function"
        self.sampler_function = sampler_function


@pytest.mark.parametrize(
    "sampler,label",
    [
        (StubSampler("euler"), "euler"),
        (StubSampler("res_multistep"), "res_multistep"),
        (_FunctionSuffixSampler("dpm_fast"), "dpm_fast"),
        (_ClassNamedSampler(), "_ClassNamedSampler"),
    ],
)
def test_sampler_label_strips_sample_prefix_and_function_suffix_or_uses_class_name(sampler, label):
    module = importlib.import_module("sampler_sigma_harvest_node")
    assert module._sampler_label(sampler) == label


@pytest.mark.parametrize("name", ["euler", "heun", "res_multistep"])
def test_harvest_runs_the_given_sampler_and_emits_its_label(name):
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    sampler = StubSampler(name)
    guider = Guider()
    text, diagnostic = _harvest(
        cls,
        guider,
        sampler,
        **{"Tolerance (Delta)": .01},
    )
    calibration = json.loads(text)

    assert len(guider.samplers) == 1 and guider.samplers[0] is sampler
    assert calibration["sampler_name"] == name
    assert calibration["schema_version"] == 2
    assert calibration["measurement_basis"] == "residual_x_minus_denoised"
    assert calibration["calibration_kind"] == "empirical_residual_fit"
    assert calibration["delta"] == .01
    assert 0 < calibration["noise_amplitude"] < 1e6
    assert -5 <= calibration["noise_decay_exponent"] < 10
    assert {"r2", "health", "report"} <= set(calibration)
    assert isinstance(diagnostic, dict) and "samples" in diagnostic

    report_lines = calibration["report"].splitlines()
    assert name in report_lines[0]
    if calibration["noise_decay_exponent"] > 0 and calibration["health"] != "invalid":
        paste_line = next(
            line for line in report_lines if line.startswith("Paste into SPEED Sampler")
        )
        assert f"same sampler: {name}" in paste_line
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

    text, _ = _harvest(cls, StreamingGuider())
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
    text, _ = _harvest(cls, Guider())
    calibration = json.loads(text)

    assert calibration["health"] == "suspect"
    assert "Do not paste this calibration into Automatic" in calibration["report"]
    assert "Paste into SPEED Sampler" not in calibration["report"]


def test_removed_harvest_tolerance_alias_fails_closed():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    with pytest.raises(TypeError, match="Unexpected Harvest option"):
        _harvest(cls, Guider(), delta=.01)


def test_harvest_rejects_unknown_model_type_instead_of_taking_the_image_path():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    with pytest.raises(ValueError, match="model_type must be one of"):
        _harvest(cls, Guider(), model_type="sdxl")


def test_harvest_reports_no_captures_instead_of_inventing_fit():
    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig

    class SilentGuider(Guider):
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            self.samplers.append(sampler)
            return latent_image

    text, diagnostic = _harvest(cls, SilentGuider(), StubSampler("heun"))
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

        text, _ = _harvest(cls, FailingGuider(), StubSampler("dpm_2"))
    else:
        monkeypatch.setattr(module, "fit_power_law", lambda *args: (_ for _ in ()).throw(ValueError(message)))
        text, _ = _harvest(cls, Guider(), StubSampler("dpm_2"))

    error = json.loads(text)
    assert error["error"] == error_kind
    assert message in error["message"]
    assert error["sampler_name"] == "dpm_2"


def test_harvest_surfaces_first_residual_capture_failure(monkeypatch, caplog):
    module = importlib.import_module("sampler_sigma_harvest_node")
    cls = module.MiniMaxH3HarvestToConfig

    def explode(*args, **kwargs):
        raise RuntimeError("residual exploded")

    monkeypatch.setattr(module, "compute_residual", explode)
    with caplog.at_level("WARNING", logger=module.__name__):
        text, _ = _harvest(cls, Guider())

    error = json.loads(text)
    assert error["error"] == "no_captures"
    assert "residual exploded" in error["message"]
    assert sum("residual capture failed" in record.message for record in caplog.records) == 1


def test_harvest_surfaces_first_profile_reduction_failure(monkeypatch, caplog):
    module = importlib.import_module("sampler_sigma_harvest_node")
    cls = module.MiniMaxH3HarvestToConfig

    def explode(*args, **kwargs):
        raise RuntimeError("profile exploded")

    monkeypatch.setattr(module, "radial_dct_power", explode)
    with caplog.at_level("WARNING", logger=module.__name__):
        text, _ = _harvest(cls, Guider())

    error = json.loads(text)
    assert error["error"] == "no_spectral_profiles"
    assert "profile exploded" in error["message"]
    assert error["n_captures"] > 0
    assert sum("spectral profile reduction failed" in record.message for record in caplog.records) == 1


class _ImageGuider(Guider):
    """Plain image latent; the model samples as ``sampling``."""

    def __init__(self, sampling):
        super().__init__()
        model = self.model_patcher.model
        self.model_patcher = type("P", (), {
            "model": model,
            "get_model_object": lambda self, name: sampling,
        })()

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        total = len(sigmas) - 1
        for step in range(total):
            if callback is not None:
                callback(step, latent_image * .5, latent_image, total)
        return latent_image


def _harvest_image(cls, guider, **kwargs):
    return cls().harvest(
        noise=Noise(),
        guider=guider,
        sampler=StubSampler(),
        sigmas=torch.linspace(1.0, .025, 20),
        latent_image={"samples": torch.randn(1, 4, 16, 16)},
        **kwargs,
    )


def test_harvest_image_runs_the_same_flow_check_as_the_sampler():
    import comfy.model_sampling

    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    eps = _ImageGuider(comfy.model_sampling.EPS())
    with pytest.raises(ValueError, match=r"flow \(CONST\) models only; this model uses EPS"):
        _harvest_image(cls, eps, model_type="image")
    assert eps.samplers == []

    text, _ = _harvest_image(cls, _ImageGuider(comfy.model_sampling.CONST()), model_type="image")
    assert json.loads(text)["model_type"] == "image"


def test_harvest_points_at_model_type_when_the_latent_does_not_match():
    import comfy.model_sampling

    cls = importlib.import_module("sampler_sigma_harvest_node").MiniMaxH3HarvestToConfig
    text, _ = _harvest_image(cls, _ImageGuider(comfy.model_sampling.CONST()))
    error = json.loads(text)
    assert error["error"] == "no_captures"
    assert "does not match model_type h3_video; use model_type image" in error["message"]
    assert "did not fire" not in error["message"]
