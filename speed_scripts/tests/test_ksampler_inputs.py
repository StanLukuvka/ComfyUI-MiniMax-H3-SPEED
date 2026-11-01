"""build_ksampler_inputs and KSamplerNoise: KSampler's noise, guider, sampler and sigmas."""

import torch

import comfy.sample
import comfy.samplers

from speed_scripts.ksampler_inputs import KSamplerNoise, build_ksampler_inputs

MODEL, SAMPLER, NOISE = object(), object(), object()
SAMPLES = torch.zeros(1, 4, 8, 8)
SIGMAS = torch.linspace(1.0, 0.0, 6)


class FakeGuider:
    """CFGGuider stand-in recording the model, conds and cfg it was given."""

    def __init__(self, model):
        self.model = model
        self.conds = self.cfg = None

    def set_conds(self, positive, negative):
        self.conds = (positive, negative)

    def set_cfg(self, cfg):
        self.cfg = cfg


class FakeKSampler:
    """KSampler stand-in recording its keyword arguments and serving fixed sigmas."""

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.sigmas = SIGMAS


def stub_comfy(monkeypatch):
    """Stand in for the comfy surface the builder uses; return what it created."""
    made = {}

    class RecordingGuider(FakeGuider):
        def __init__(self, model):
            super().__init__(model)
            made["guider"] = self

    class RecordingKSampler(FakeKSampler):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            made["ksampler"] = self

    def sampler_object(name):
        made["sampler_name"] = name
        return SAMPLER

    # CFGGuider and sampler_object are absent from the conftest's comfy stub, so they
    # must be created here; KSampler exists there and a missing one fails loudly.
    monkeypatch.setattr(comfy.samplers, "CFGGuider", RecordingGuider, raising=False)
    monkeypatch.setattr(comfy.samplers, "KSampler", RecordingKSampler)
    monkeypatch.setattr(comfy.samplers, "sampler_object", sampler_object, raising=False)
    return made


def test_build_ksampler_inputs_builds_the_cfg_guider_and_sampler(monkeypatch):
    """The guider is a CFGGuider on the model with the given conds and cfg; the sampler is sampler_object(name)."""
    made = stub_comfy(monkeypatch)
    noise, guider, sampler, _ = build_ksampler_inputs(
        MODEL, 42, 20, 6.5, "euler", "karras", "pos", "neg",
    )
    assert guider is made["guider"] and isinstance(guider, FakeGuider)
    assert guider.model is MODEL
    assert guider.conds == ("pos", "neg")
    assert guider.cfg == 6.5
    assert sampler is SAMPLER and made["sampler_name"] == "euler"
    assert noise.seed == 42
    # The default denoise is KSampler's full-strength 1.0.
    assert made["ksampler"].kwargs["denoise"] == 1.0


def test_build_ksampler_inputs_takes_sigmas_from_a_cpu_ksampler(monkeypatch):
    """Sigmas come from KSampler(model, steps, device="cpu", sampler, scheduler, denoise) with denoise passed through."""
    made = stub_comfy(monkeypatch)
    _, _, _, sigmas = build_ksampler_inputs(
        MODEL, 42, 25, 6.5, "dpm_2", "beta", "pos", "neg", denoise=0.6,
    )
    assert sigmas is SIGMAS
    assert made["ksampler"].model is MODEL
    assert made["ksampler"].kwargs == {
        "steps": 25, "device": "cpu", "sampler": "dpm_2", "scheduler": "beta", "denoise": 0.6,
    }


def test_ksampler_noise_keeps_its_seed():
    """KSamplerNoise is ComfyUI's RandomNoise: the seed travels on the object."""
    assert KSamplerNoise(123).seed == 123


def test_ksampler_noise_prepares_noise_with_samples_seed_and_batch_index(monkeypatch):
    """generate_noise hands samples, seed and batch_index to comfy.sample.prepare_noise."""
    calls = []

    def fake_prepare_noise(samples, seed, batch_inds):
        calls.append((samples, seed, batch_inds))
        return NOISE

    monkeypatch.setattr(comfy.sample, "prepare_noise", fake_prepare_noise, raising=False)
    noise = KSamplerNoise(7)
    # batch_index is None when the latent dict has no such key, and passes through when present.
    assert noise.generate_noise({"samples": SAMPLES}) is NOISE
    assert noise.generate_noise({"samples": SAMPLES, "batch_index": [3]}) is NOISE
    assert calls == [(SAMPLES, 7, None), (SAMPLES, 7, [3])]
