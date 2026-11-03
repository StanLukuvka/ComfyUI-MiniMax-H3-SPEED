"""KSampler Automatic and Manual nodes: build the KSampler inputs, then run the Advanced body."""

import pytest
import torch

import ksampler_node
import ksampler_node_manual
import sampler_node
import sampler_node_manual

SIGMAS = torch.linspace(1.0, 0.0, 11)


class BuilderRecorder:
    """Fake build_ksampler_inputs: records the call and returns sentinel inputs."""

    def __init__(self, sigmas):
        self.calls = []
        self.noise, self.guider, self.sampler = object(), object(), object()
        self.sigmas = sigmas

    def __call__(self, model, seed, steps, cfg, sampler_name, scheduler,
                 positive, negative, denoise):
        self.calls.append({
            "model": model, "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler_name, "scheduler": scheduler,
            "positive": positive, "negative": negative, "denoise": denoise,
        })
        return self.noise, self.guider, self.sampler, self.sigmas


def patch_pipeline(monkeypatch, advanced_module, result):
    """Patch run_speed_pipeline where the Advanced twin calls it; return the capture."""
    captured = {}

    def fake_pipeline(noise, guider, sigmas, latent_image, config, **kwargs):
        captured.update(noise=noise, guider=guider, sigmas=sigmas,
                        latent_image=latent_image, config=config, kwargs=kwargs)
        return result

    monkeypatch.setattr(advanced_module, "run_speed_pipeline", fake_pipeline)
    return captured


def automatic_inputs(**overrides):
    base = dict(
        seed=7, steps=20, cfg=4.5, sampler_name="dpm_2", scheduler="karras",
        denoise=0.75, stages=4,
        noise_policy="coupled_full_grid", noise_amplitude=12.105,
        noise_decay_exponent=0.773, seed_offset=777,
    )
    base.update(overrides)
    return base


def manual_inputs(**overrides):
    base = dict(
        seed=9, steps=10, cfg=8.0, sampler_name="euler", scheduler="normal",
        denoise=1.0, noise_policy="coupled_full_grid",
        seed_offset=55, ratio_mode="steps",
        transition_goal_1=3, transition_resolution_1=0.25,
        transition_goal_2=5, transition_resolution_2=0.5,
        transition_goal_3=8, transition_resolution_3=0.75,
        transition_goal_4=15, transition_resolution_4=1.0,
    )
    base.update(overrides)
    return base


def test_ksampler_builds_inputs_and_runs_the_advanced_automatic_body(monkeypatch):
    recorder = BuilderRecorder(SIGMAS)
    monkeypatch.setattr(ksampler_node, "build_ksampler_inputs", recorder)
    captured = patch_pipeline(monkeypatch, sampler_node, ("out", "denoised"))
    model, positive, negative = object(), object(), object()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    result = ksampler_node.SPEEDKSampler().sample(
        model=model, positive=positive, negative=negative, latent_image=latent,
        **automatic_inputs(**{"Tolerance (Delta)": 0.009}),
    )

    assert recorder.calls == [{
        "model": model, "seed": 7, "steps": 20, "cfg": 4.5,
        "sampler_name": "dpm_2", "scheduler": "karras",
        "positive": positive, "negative": negative, "denoise": 0.75,
    }]
    assert result == ("out", "denoised")
    # The built noise/guider/sampler/sigmas flow straight into the pipeline.
    assert captured["noise"] is recorder.noise
    assert captured["guider"] is recorder.guider
    assert captured["sigmas"] is recorder.sigmas
    assert captured["kwargs"]["sampler"] is recorder.sampler
    assert captured["latent_image"] is latent
    assert captured["kwargs"]["model_type"] == "flow_image"
    assert captured["kwargs"]["disable_pbar"] is False
    # The SPEED widgets reach the config exactly as in the Advanced node.
    config = captured["config"]
    assert config.scales == pytest.approx((0.25, 0.5, 0.75, 1.0))
    assert (config.transition_mode, config.transition_steps) == ("delta_custom", ())
    assert config.noise_policy == "coupled_full_grid"
    assert config.delta == 0.009
    assert (config.noise_amplitude, config.noise_decay_exponent) == (12.105, 0.773)
    assert config.transition_seed_offset == 777


def test_ksampler_automatic_defaults_a_missing_tolerance_to_delta_0005(monkeypatch):
    recorder = BuilderRecorder(SIGMAS)
    monkeypatch.setattr(ksampler_node, "build_ksampler_inputs", recorder)
    captured = patch_pipeline(monkeypatch, sampler_node, ("out", "denoised"))

    ksampler_node.SPEEDKSampler().sample(
        model=object(), positive=object(), negative=object(),
        latent_image={"samples": torch.zeros(1, 4, 8, 8)}, **automatic_inputs(),
    )
    assert captured["config"].delta == 0.005


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"stages": 5}, "stages must be exactly 2, 3, or 4"),
        ({"noise_amplitude": 0.0}, "not set .* Run Sigma Harvest"),
        ({"noise_decay_exponent": 0.0}, "not set .* Run Sigma Harvest"),
        ({"removed_option": "old"}, "Unexpected Automatic option"),
    ],
)
def test_ksampler_automatic_rejects_bad_inputs_like_the_advanced_node(monkeypatch, kwargs, error):
    recorder = BuilderRecorder(SIGMAS)
    monkeypatch.setattr(ksampler_node, "build_ksampler_inputs", recorder)
    patch_pipeline(monkeypatch, sampler_node, ("out", "denoised"))
    with pytest.raises((TypeError, ValueError), match=error):
        ksampler_node.SPEEDKSampler().sample(
            model=object(), positive=object(), negative=object(),
            latent_image={"samples": torch.zeros(1, 4, 8, 8)},
            **automatic_inputs(**kwargs),
        )


def test_ksampler_manual_builds_inputs_and_runs_the_advanced_manual_body(monkeypatch):
    recorder = BuilderRecorder(SIGMAS)
    monkeypatch.setattr(ksampler_node_manual, "build_ksampler_inputs", recorder)
    captured = patch_pipeline(monkeypatch, sampler_node_manual, ("mout", "mden"))
    model, positive, negative = object(), object(), object()
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    result = ksampler_node_manual.SPEEDKSamplerManual().sample(
        model=model, positive=positive, negative=negative, latent_image=latent,
        **manual_inputs(ratio_mode="ratio", transition_goal_1=0.3,
                        transition_goal_2=0.6, transition_goal_3=0),
    )

    assert recorder.calls == [{
        "model": model, "seed": 9, "steps": 10, "cfg": 8.0,
        "sampler_name": "euler", "scheduler": "normal",
        "positive": positive, "negative": negative, "denoise": 1.0,
    }]
    assert result == ("mout", "mden")
    assert captured["noise"] is recorder.noise
    assert captured["guider"] is recorder.guider
    assert captured["sigmas"] is recorder.sigmas
    assert captured["kwargs"]["sampler"] is recorder.sampler
    assert captured["latent_image"] is latent
    assert captured["kwargs"]["model_type"] == "flow_image"
    # The SPEED schedule widgets reach the config exactly as in the Advanced node.
    config = captured["config"]
    assert config.scales == pytest.approx((0.25, 0.5, 1.0))
    assert config.transition_steps == (3, 6)
    assert config.transition_mode == "explicit"
    assert config.noise_policy == "coupled_full_grid"
    assert config.transition_seed_offset == 55


def test_ksampler_manual_rejects_bad_schedules_like_the_advanced_node(monkeypatch):
    recorder = BuilderRecorder(SIGMAS)
    monkeypatch.setattr(ksampler_node_manual, "build_ksampler_inputs", recorder)
    patch_pipeline(monkeypatch, sampler_node_manual, ("mout", "mden"))
    with pytest.raises(ValueError, match="whole step index"):
        ksampler_node_manual.SPEEDKSamplerManual().sample(
            model=object(), positive=object(), negative=object(),
            latent_image={"samples": torch.zeros(1, 4, 8, 8)},
            **manual_inputs(transition_goal_2=5.5),
        )


@pytest.mark.parametrize("short_sigmas", [torch.zeros(0), torch.tensor([0.5])],
                         ids=["empty", "single"])
@pytest.mark.parametrize(
    "node_module,node_cls,advanced_module,inputs",
    [
        (ksampler_node, ksampler_node.SPEEDKSampler, sampler_node,
         automatic_inputs(denoise=0.0, noise_amplitude=0.0, noise_decay_exponent=0.0)),
        (ksampler_node_manual, ksampler_node_manual.SPEEDKSamplerManual,
         sampler_node_manual, manual_inputs(denoise=0.0)),
    ],
    ids=["automatic", "manual"],
)
def test_fewer_than_two_sigmas_returns_the_latent_unchanged(
        monkeypatch, short_sigmas, node_module, node_cls, advanced_module, inputs):
    recorder = BuilderRecorder(short_sigmas)
    monkeypatch.setattr(node_module, "build_ksampler_inputs", recorder)
    ran = []
    monkeypatch.setattr(advanced_module, "run_speed_pipeline",
                        lambda *args, **kwargs: ran.append((args, kwargs)))
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    output, denoised = node_cls().sample(
        model=object(), positive=object(), negative=object(),
        latent_image=latent, **inputs,
    )

    # The latent comes back unchanged in the normal (output, denoised) shape,
    # without validation or the pipeline (denoise 0 skips even unset A/beta).
    assert output is latent and denoised is latent
    assert recorder.calls[0]["denoise"] == 0.0
    assert ran == []
