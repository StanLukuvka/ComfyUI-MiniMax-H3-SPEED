"""Sigma Harvest node: native run with the connected sampler, calibration JSON, and error JSONs."""

import importlib
import json

import pytest
import torch

from conftest import EchoNoise, H3Model, NestedTensor, Patcher, StubSampler, model_sampling

harvest_module = importlib.import_module("sampler_sigma_harvest_node")


class HarvestGuider:
    """Native-sampler stand-in: Euler steps on a denoiser that predicts half the state.

    Fires the callback after every step (unless ``silent``), then ``on_step(step)``.
    """

    def __init__(self, sampling=None, on_step=None, silent=False):
        self.model_patcher = Patcher(H3Model(), model_sampling=sampling or model_sampling())
        self.on_step = on_step
        self.silent = silent
        self.samplers = []

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.samplers.append(sampler)
        if self.silent:
            return latent_image
        nested = getattr(latent_image, "is_nested", False)
        state = (latent_image.unbind()[0] if nested else latent_image).float()
        total = len(sigmas) - 1
        for step in range(total):
            sigma = float(sigmas[step])
            denoised = state * .5
            state = state + (state - denoised) / sigma * (float(sigmas[step + 1]) - sigma)
            out = NestedTensor([state, torch.zeros(1, 1, 2, 44)]) if nested else state
            den = NestedTensor([denoised, torch.zeros(1, 1, 2, 44)]) if nested else denoised
            callback(step, den, out, total)
            if self.on_step is not None:
                self.on_step(step)
        return out


def _red_noise(*shape):
    """Seeded noise integrated over H and W, so its power falls off with frequency."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(0)).cumsum(-1).cumsum(-2)


def _h3_latent():
    return {"samples": NestedTensor([_red_noise(1, 4, 8, 16, 16), torch.zeros(1, 1, 2, 44)])}


def _image_latent():
    return {"samples": _red_noise(1, 4, 16, 16)}


def _harvest(guider=None, sampler=None, latent=None, **kwargs):
    kwargs.setdefault("model_type", "h3_video")
    return harvest_module.SPEEDSigmaHarvest().harvest(
        noise=EchoNoise(),
        guider=guider or HarvestGuider(),
        sampler=sampler or StubSampler(),
        sigmas=torch.linspace(1.0, .025, 20),
        latent_image=latent or _h3_latent(),
        **kwargs,
    )


class _ClassNamedSampler:
    """A SAMPLER without ``sampler_function``; labelled by its class name."""


def _function_suffix_sampler(name):
    """A SAMPLER whose function is named ``<name>_function`` (e.g. dpm_fast)."""
    sampler = StubSampler()
    sampler.sampler_function.__name__ = f"{name}_function"
    return sampler


@pytest.mark.parametrize(
    "sampler,name,model_type,latent",
    [
        (StubSampler("euler"), "euler", "h3_video", _h3_latent),
        (StubSampler("res_multistep"), "res_multistep", "h3_video", _h3_latent),
        (_function_suffix_sampler("dpm_fast"), "dpm_fast", "h3_video", _h3_latent),
        (_ClassNamedSampler(), "_ClassNamedSampler", "h3_video", _h3_latent),
        (StubSampler("euler"), "euler", "flow_image", _image_latent),
    ],
    ids=["euler", "res_multistep", "function_suffix", "class_name", "flow_image"],
)
def test_harvest_runs_the_given_sampler_once_and_emits_its_calibration(sampler, name, model_type, latent):
    guider = HarvestGuider()
    text, diagnostic = _harvest(guider, sampler, latent(), model_type=model_type, **{"Tolerance (Delta)": .01})
    calibration = json.loads(text)

    assert guider.samplers == [sampler]
    assert {key: calibration[key] for key in (
        "schema_version", "sampler_name", "model_type", "delta", "measurement_basis", "calibration_kind",
    )} == {
        "schema_version": 2, "sampler_name": name, "model_type": model_type, "delta": .01,
        "measurement_basis": "residual_x_minus_denoised", "calibration_kind": "empirical_residual_fit",
    }
    assert 0 < calibration["noise_amplitude"] < 1e6
    assert 0 < calibration["noise_decay_exponent"] < 10
    assert {"r2", "health"} <= set(calibration)
    assert "samples" in diagnostic

    report = calibration["report"].splitlines()
    assert name in report[0]
    paste = next(line for line in report if line.startswith("Paste into SPEED Sampler"))
    assert f"same sampler: {name}" in paste
    assert f"noise_amplitude={calibration['noise_amplitude']:.4f}" in paste
    assert f"noise_decay_exponent={calibration['noise_decay_exponent']:.4f}" in paste
    assert "Tolerance (Delta)=0.010" in paste


def test_harvest_reduces_each_residual_inside_the_callback(monkeypatch):
    reduced = []
    original = harvest_module.radial_dct_power

    def recording(residual):
        reduced.append(residual.shape)
        return original(residual)

    def check(step):
        assert len(reduced) == step + 1  # reduced before the sampler's next step

    monkeypatch.setattr(harvest_module, "radial_dct_power", recording)
    text, _ = _harvest(HarvestGuider(on_step=check))
    assert "error" not in json.loads(text)
    assert len(reduced) == 19


@pytest.mark.parametrize(
    "beta,r2,health,paste_ready",
    [
        (0.8, 0.9, "good", True),
        (0.8, 0.5, "fair", True),
        (0.8, 0.1, "weak", True),
        (-0.5, 0.2, "suspect", False),
        (float("nan"), 0.9, "invalid", False),
    ],
)
def test_fit_health_decides_whether_the_report_is_paste_ready(monkeypatch, beta, r2, health, paste_ready):
    monkeypatch.setattr(
        harvest_module, "fit_power_law", lambda *args: {"A": 1.0, "beta": beta, "r_squared": r2, "n_bins": 8}
    )
    calibration = json.loads(_harvest()[0])
    assert calibration["health"] == health
    assert ("Paste into SPEED Sampler" in calibration["report"]) is paste_ready
    assert ("Do not paste this calibration into Automatic" in calibration["report"]) is not paste_ready
    assert ("WARNING" in calibration["report"]) is (health not in ("good", "fair"))


def _explode(error=RuntimeError):
    def explode(*args, **kwargs):
        raise error('bad "quote"\nnext line\\slash')

    return explode


@pytest.mark.parametrize(
    "patch,guider,inputs,error,message,log_text",
    [
        ({}, HarvestGuider(silent=True), {}, "no_captures", "callback did not fire", None),
        (None, None, {}, "harvest_failed", 'bad "quote"\nnext line\\slash', None),
        # Harvest reports a fit's ValueError; the error text survives JSON escaping.
        ({"fit_power_law": _explode(ValueError)}, None, {}, "fit_failed", 'bad "quote"\nnext line\\slash', None),
        # The first failure of each capture step is surfaced once, not swallowed.
        ({"compute_residual": _explode()}, None, {}, "no_captures", "Residual capture failed: bad", "residual capture failed"),
        ({"radial_dct_power": _explode()}, None, {}, "no_spectral_profiles", "Spectral profile reduction failed: bad",
         "spectral profile reduction failed"),
        # A latent that does not match model_type points at the right setting.
        ({}, None, {"latent": _image_latent()}, "no_captures",
         "does not match model_type h3_video; use model_type flow_image", None),
        # A plain 5-D latent (Krea 2's [B,C,1,H,W]) is not H3's nested video/audio latent.
        ({}, None, {"latent": {"samples": _red_noise(1, 4, 1, 16, 16)}}, "no_captures",
         "does not match model_type h3_video; use model_type flow_image", None),
        ({}, None, {"model_type": "flow_image"}, "no_captures",
         "does not match model_type flow_image; use model_type h3_video", None),
    ],
    ids=["silent_sampler", "sampler_failed", "fit_failed", "residual_failed", "profile_failed",
         "image_latent_as_h3", "five_d_image_latent_as_h3", "h3_latent_as_image"],
)
def test_harvest_returns_an_error_json_instead_of_a_calibration(
    monkeypatch, caplog, patch, guider, inputs, error, message, log_text
):
    guider = guider or HarvestGuider()
    if patch is None:
        guider.sample = _explode()
    for name, replacement in (patch or {}).items():
        monkeypatch.setattr(harvest_module, name, replacement)
    with caplog.at_level("WARNING", logger=harvest_module.__name__):
        text, diagnostic = _harvest(guider, StubSampler("dpm_2"), **inputs)

    result = json.loads(text)
    assert (result["error"], result["sampler_name"]) == (error, "dpm_2")
    assert message in result["message"]
    assert "samples" in diagnostic
    if log_text:
        assert sum(log_text in record.message for record in caplog.records) == 1


@pytest.mark.parametrize(
    "inputs,error,match",
    [
        ({"delta": .01}, TypeError, "Unexpected Harvest option"),  # the removed Tolerance alias
        ({"model_type": "sdxl"}, ValueError, "model_type must be one of"),
        # The image path runs the same flow-model check as the sampler, before sampling.
        ({"model_type": "flow_image", "latent": _image_latent(),
          "guider": HarvestGuider(sampling=model_sampling("EPS"))},
         ValueError, r"flow \(CONST\) models only; this model uses EPS"),
    ],
    ids=["removed_delta_alias", "unknown_model_type", "non_flow_image_model"],
)
def test_harvest_rejects_bad_inputs(inputs, error, match):
    guider = inputs.pop("guider", HarvestGuider())
    with pytest.raises(error, match=match):
        _harvest(guider, **inputs)
    assert guider.samplers == []
