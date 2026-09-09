"""One end-to-end contract for native-Euler Sigma Trace telemetry."""

import importlib
import json
import sys
from types import ModuleType

import pytest
import torch

from conftest import make_nested


class Noise:
    seed = 73

    def __init__(self, samples):
        self.samples = samples
        self.calls = 0

    def generate_noise(self, latent):
        self.calls += 1
        return self.samples


class EulerGuider:
    def __init__(self):
        self.model_patcher = object()
        self.calls = 0
        self.predictions = []
        self.result = None

    def sample(
        self,
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=None,
    ):
        self.calls += 1
        video, audio = noise.unbind()
        states = [video.clone(), audio.clone()]
        count = len(sigmas) - 1
        for step in range(count):
            x0 = make_nested(
                states[0].tanh() * (step + 1) / count,
                states[1].tanh() * (step + 1) / count,
            )
            self.predictions.append(x0)
            if callback is not None:
                callback(step, x0, make_nested(*states), count)
            sigma = float(sigmas[step])
            sigma_next = float(sigmas[step + 1])
            states = [
                state + ((state - pred) / sigma) * (sigma_next - sigma)
                for state, pred in zip(states, x0.unbind())
            ]
        self.result = make_nested(*states)
        return self.result


def test_sigma_trace_runs_one_native_pass_and_records_x0_trajectory(monkeypatch):
    previews = []
    preview_module = ModuleType("latent_preview")

    def prepare_callback(patcher, steps, x0_output_dict=None):
        def callback(step, x0, x, total_steps):
            previews.append((step, x0, total_steps))
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0
        return callback

    preview_module.prepare_callback = prepare_callback
    monkeypatch.setitem(sys.modules, "latent_preview", preview_module)

    module = importlib.import_module("sampler_sigma_trace_node")
    module = importlib.reload(module)
    cls = module.MiniMaxH3SigmaTrace

    video = torch.linspace(-1, 1, 24 * 3 * 6 * 8).reshape(1, 24, 3, 6, 8)
    audio = torch.linspace(-.5, .5, 32 * 9).reshape(1, 32, 9)
    original_samples = make_nested(torch.zeros_like(video), torch.zeros_like(audio))
    mask = torch.ones(1, 1, 3, 6, 8)
    latent = {"samples": original_samples, "noise_mask": mask, "metadata": "keep"}
    noise = Noise(make_nested(video, audio))
    guider = EulerGuider()
    sigmas = torch.tensor([1.0, .7, .25, 0.0])

    text, output = cls().trace(noise, guider, sigmas, latent)
    document = json.loads(text)

    assert noise.calls == guider.calls == 1
    assert document["sampler"] == "euler"
    assert document["measured_tensor"] == "denoised_x0_video"
    assert document["callback_count"] == document["expected_steps"] == 3
    assert document["complete"] is True
    assert [r["step_index"] for r in document["records"]] == [0, 1, 2]
    assert [r["sigma"] for r in document["records"]] == pytest.approx(sigmas[:-1].tolist())
    assert [r["sigma_next"] for r in document["records"]] == pytest.approx(sigmas[1:].tolist())

    for step, record in enumerate(document["records"]):
        predicted_video = guider.predictions[step].unbind()[0]
        assert record["status"] == "ok"
        assert record["signal"]["rms"] == pytest.approx(
            predicted_video.square().mean().sqrt().item(),
            rel=1e-5,
        )
        assert set(record["spatial_dct"]["bands"]) == {"low", "mid", "high"}
        assert record["temporal_dct"]["available"] is True

    assert [step for step, _x0, _total in previews] == [0, 1, 2]
    assert output is not latent
    assert output["samples"] is guider.result
    assert output["noise_mask"] is mask
    assert output["metadata"] == "keep"
    assert latent["samples"] is original_samples
    assert not {"A", "beta", "calibration"} & document.keys()
