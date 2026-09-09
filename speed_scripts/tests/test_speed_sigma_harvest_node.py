"""Integration contracts for Continuous SPEED Sigma Harvest."""

import importlib
import json

import pytest
import torch

from conftest import make_fake_noise, make_latent


SIGMAS = torch.tensor([1.0, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.0])
CALIBRATION = {
    "delta": .01,
    "noise_amplitude": 7.394,
    "noise_decay_exponent": .62,
}


class SpectralGuider:
    """Echo guider that emits deterministic non-zero x0 predictions per step."""

    def __init__(self):
        self.sigma_calls = []
        self.model_patcher = type(
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
        self.conds = {}

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.sigma_calls.append([float(s) for s in sigmas])
        count = len(sigmas) - 1
        for step in range(count):
            streams = []
            for stream in latent_image.unbind():
                generator = torch.Generator().manual_seed(1000 + step)
                pattern = torch.rand(stream.shape, generator=generator) * 2.0 - 1.0
                streams.append(pattern * ((step + 1) / max(count, 1)))
            x0 = type(
                "NestedX0",
                (),
                {"is_nested": True, "unbind": lambda self, parts=streams: list(parts)},
            )()
            if callback is not None:
                callback(step, x0, latent_image, count)
        return latent_image


def _run(*, measurement_mode="both", analysis_stride=1, latent=None, **overrides):
    mod = importlib.import_module("sampler_speed_sigma_harvest_node")
    guider = SpectralGuider()
    kwargs = dict(CALIBRATION)
    kwargs.update(overrides)
    text, output, denoised = mod.MiniMaxH3SPEEDSigmaHarvest().sample(
        noise=make_fake_noise(),
        guider=guider,
        sigmas=SIGMAS,
        latent_image=latent or make_latent(h=72, w=80, metadata="keep"),
        stages=4,
        noise_policy="direct_coarse",
        measurement_mode=measurement_mode,
        analysis_stride=analysis_stride,
        **kwargs,
    )
    return text, output, denoised, guider


def _strict_load(text):
    def reject(value):
        raise AssertionError(f"non-JSON numeric constant: {value}")
    return json.loads(text, parse_constant=reject)


def test_node_public_surface_matches_automatic_generation_inputs():
    harvest = importlib.import_module(
        "sampler_speed_sigma_harvest_node"
    ).MiniMaxH3SPEEDSigmaHarvest
    automatic = importlib.import_module("sampler_node").MiniMaxH3SPEEDSampler

    assert harvest.RETURN_TYPES == ("STRING", "LATENT", "LATENT")
    assert harvest.RETURN_NAMES == ("harvest_json", "output", "denoised_output")
    harvest_required = harvest.INPUT_TYPES()["required"]
    auto_required = automatic.INPUT_TYPES()["required"]
    for name, spec in auto_required.items():
        assert harvest_required[name] == spec
    assert {"measurement_mode", "analysis_stride", "smoothing_alpha",
            "boundary_band_half_width", "store_radial_profiles"} <= set(harvest_required)


def test_one_speed_run_emits_complete_strict_telemetry_and_outputs():
    text, output, denoised, guider = _run()
    document = _strict_load(text)

    assert len(guider.sigma_calls) == 4
    assert document["schema_version"] == 1
    assert document["mode"] == "observational"
    assert document["adaptive_control"] is False
    assert document["measurement_bases"] == [
        "denoised_x0_video",
        "residual_x_minus_x0_video",
    ]
    assert [record["callback_index"] for record in document["records"]] == list(range(10))
    assert [record["global_schedule_index"] for record in document["records"]] == list(range(10))
    assert len(document["transitions"]) == 3

    first = document["records"][0]
    assert first["x0_signal"]["fit"]["status"] == "ok"
    assert first["residual"]["fit"]["status"] == "ok"
    boundary = first["x0_signal"]["current_boundary"]
    assert boundary["direct_available"] is True
    assert boundary["power_point"] > 0
    assert boundary["activation_threshold_point"] is not None
    assert first["x0_signal"]["fit_predictions"]
    assert document["summary"]

    assert output["metadata"] == "keep"
    assert denoised["metadata"] == "keep"
    assert "NaN" not in text and "Infinity" not in text


@pytest.mark.parametrize(
    "mode,present,absent",
    [
        ("x0_only", "x0_signal", "residual"),
        ("residual_only", "residual", "x0_signal"),
    ],
)
def test_measurement_mode_and_stride_control_work(mode, present, absent):
    text, _output, _denoised, _guider = _run(
        measurement_mode=mode,
        analysis_stride=2,
    )
    document = _strict_load(text)
    measured = [
        record for record in document["records"]
        if present in record
    ]
    assert [record["callback_index"] for record in measured] == [0, 2, 4, 6, 8]
    assert all(absent not in record for record in measured)
    skeleton = next(record for record in document["records"] if record["callback_index"] == 1)
    assert present not in skeleton and absent not in skeleton
