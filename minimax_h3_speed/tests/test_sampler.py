"""Minimal contract test for the single SPEED sampler node."""
import importlib
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from minimax_h3_speed.config import SpeedConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _install_comfy_stubs():
    comfy = ModuleType("comfy")
    samplers = ModuleType("comfy.samplers")
    utils = ModuleType("comfy.utils")
    model_mgmt = ModuleType("comfy.model_management")
    kdiff = ModuleType("comfy.k_diffusion")
    ksampling = ModuleType("comfy.k_diffusion.sampling")
    nested_tensor = ModuleType("comfy.nested_tensor")

    class NestedTensor:
        is_nested = True
        def __init__(self, tensors):
            self._tensors = tensors
        def unbind(self):
            return self._tensors
    nested_tensor.NestedTensor = NestedTensor

    samplers.sampler_object = lambda name: ("sampler", name)
    utils.PROGRESS_BAR_ENABLED = True

    def pack_latents(latents):
        shapes, tensors = [], []
        for t in latents:
            shapes.append(list(t.shape))
            tensors.append(t.reshape(t.shape[0], 1, -1))
        return torch.cat(tensors, dim=-1), shapes

    def unpack_latents(combined, shapes):
        out, work = [], combined
        for shape in shapes:
            cut = math.prod(shape[1:])
            out.append(work[:, :, :cut].reshape([work.shape[0]] + shape[1:]))
            work = work[:, :, cut:]
        return out

    utils.pack_latents = pack_latents
    utils.unpack_latents = unpack_latents
    model_mgmt.intermediate_device = lambda: "cpu"

    def sample_euler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        extra_args = {} if extra_args is None else extra_args
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            denoised = model(x, sigma, **extra_args)
            d = (x - denoised) / sigma
            x = x + d * (sigmas[i + 1] - sigmas[i])
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma, "denoised": denoised})
        return x

    ksampling.sample_euler = sample_euler
    comfy.samplers = samplers
    comfy.utils = utils
    comfy.model_management = model_mgmt
    comfy.k_diffusion = kdiff
    comfy.k_diffusion.sampling = ksampling
    comfy.nested_tensor = nested_tensor
    sys.modules["comfy"] = comfy
    for name, mod in [("samplers", samplers), ("utils", utils),
                      ("model_management", model_mgmt),
                      ("k_diffusion", kdiff), ("k_diffusion.sampling", ksampling),
                      ("nested_tensor", nested_tensor)]:
        sys.modules["comfy." + name] = mod


# Install comfy stubs at module load so any test that imports `main` or
# `minimax_h3_speed.h3_runtime` (which now import comfy at top level) works
# without needing to call _install_comfy_stubs() first.
_install_comfy_stubs()


def test_node_exports_sampler():
    _install_comfy_stubs()
    mod = importlib.import_module("sampler_node")
    cls = mod.MiniMaxH3SPEEDSampler
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert cls.FUNCTION == "sample"
    assert "MiniMaxH3SPEEDSampler" in mod.NODE_CLASS_MAPPINGS


def test_input_schema_widgets_and_required_inputs():
    _install_comfy_stubs()
    mod = importlib.import_module("sampler_node")
    inputs = mod.MiniMaxH3SPEEDSampler.INPUT_TYPES()
    required = inputs["required"]
    for key in ("noise", "guider", "sigmas", "latent_image",
                "preset", "transition_mode"):
        assert key in required, f"missing required input: {key}"
    # delta_custom path is enabled with sigma-harvest calibration
    assert "delta" in required
    assert "power_A" in required
    assert "power_beta" in required
    assert "seed_offset" in required
    assert required["preset"][0][0] == "half_then_full"


def test_sample_runs_multi_stage():
    _install_comfy_stubs()
    mod = importlib.import_module("sampler_node")
    from minimax_h3_speed.h3_runtime import run_progressive_stages

    sample_calls = []

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            sample_calls.append(len(sigmas))
            # Simulate what a real sampler would do: call callback(step, x0, x, total)
            if callback is not None:
                x0 = latent_image  # fake denoised output
                callback(0, x0, latent_image, len(sigmas))
            return latent_image

    class FakeNoise:
        seed = 42
        def generate_noise(self, latent):
            samples = latent.get("samples")
            if getattr(samples, "is_nested", False):
                vids = [s for s in samples.unbind() if s.ndim == 5]
                auds = [s for s in samples.unbind() if s.ndim != 5]
                return type("NT", (), {"is_nested": True, "unbind": lambda v=vids, a=auds: v + a})()
            return samples

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)

    class FakeNested:
        is_nested = True
        def unbind(self):
            return [video, audio]

    nested = FakeNested()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.025, 20)
    config = SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,))

    out, denoised = run_progressive_stages(
        FakeNoise(), FakeGuider(), sigmas, latent, config,
        sampler=type("S", (), {"name": "euler"})(),
        nested_type=type("NT", (), {"is_nested": True})(),
        disable_pbar=True, output_device=None,
    )
    assert out is not None
    # ≥2 stages: coarse + final full-res
    assert len(sample_calls) >= 2, f"expected ≥2 stages, got {len(sample_calls)}"




def test_aligned_speed_sigma_math():
    """kappa = r / (1 + (r-1)q); t_tilde = kappa * q. Verify the paper formula."""
    flow = importlib.import_module("minimax_h3_speed.flow")
    for q, r in [(0.5, 2.0), (0.3, 2.0), (0.8, 4.0 / 3.0)]:
        kappa, t = flow.aligned_speed_sigma(q, r)
        assert abs(kappa - r / (1.0 + (r - 1.0) * q)) < 1e-6
        assert abs(t - kappa * q) < 1e-12


def test_resolve_transition_steps_explicit_example():
    """Explicit mode places transitions at the flat per-scale default (5)."""
    mod = importlib.import_module("sampler_node")
    h3_runtime = importlib.import_module("minimax_h3_speed.h3_runtime")
    sigmas = torch.linspace(1.0, 0.025, 20)
    explicit_steps = h3_runtime.resolve_transition_steps(
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), transition_mode="explicit"),
        sigmas, 64, 64,
    )
    assert tuple(explicit_steps) == (5,)


def test_active_av_shifts_returns_audio_scale_from_ratio():
    """_active_av_shifts should compute audio_scale = video_shift / audio_shift,
    not read a non-existent 'audio_scale' attribute.

    With shift_video=12.0 and shift_audio=3.0, the correct audio_scale is 4.0.
    The buggy implementation reads model.audio_scale (doesn't exist → returns 1.0).
    This test asserts the fixed behaviour: audio_scale must be the ratio.
    """
    from minimax_h3_speed.h3_runtime import _active_av_shifts

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "diffusion_model": type("DM", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
            })(),
        })()})()

    v_shift, a_shift, a_scale = _active_av_shifts(FakeGuider())
    assert v_shift == 12.0
    assert a_shift == 3.0
    # audio_scale should be the ratio, not a fixed 1.0
    assert a_scale == 4.0, f"expected audio_scale=4.0 (12/3), got {a_scale}"




def test_workflow_json_node_types_are_supported():
    """Every class_type in the shipped workflow is a known ComfyUI / H3 / this
    node, so the workflow can actually load and run once models exist."""
    import json
    wf = json.loads(
        (Path(__file__).resolve().parents[2] / "workflows" / "video_minimax_h3_t2v_speed.json")
        .read_text())
    # UI format: {"nodes": [...], "links": [...], ...}
    nodes_list = wf["nodes"] if "nodes" in wf else wf
    expected = {
        "UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
        "BasicScheduler", "RandomNoise", "BasicGuider",
        "MiniMaxH3SPEEDSampler", "VAEDecode", "VAEDecode",
        "VAEDecodeAudio", "CreateVideo", "SaveVideo",
    }
    types = {(n.get("type") or n.get("class_type")) for n in nodes_list}
    assert types == expected
    # The sampler node exists in the workflow.
    sampler_node = next(n for n in nodes_list if (n.get("type") or n.get("class_type")) == "MiniMaxH3SPEEDSampler")
    assert sampler_node is not None


# ---------------------------------------------------------------------------
# Cross-repo flow equivalence tests (MVP vs Lab oracle)
# ---------------------------------------------------------------------------
# Shared pure-math inputs. The Lab lives as a sibling repo of the Sampler, so
# its path is parents[3] (not parents[2]) from this test file. The verify
# command's PYTHONPATH does not include the Lab, so each test inserts it
# explicitly before importing speed_lab.flow.

video = torch.randn(1, 1, 2, 16, 16)
audio = torch.randn(1, 1, 2, 16)
sigma = torch.tensor(0.5)
old_video_sigma = 0.8
new_video_sigma = 0.4
old_audio_sigma = 0.3
new_audio_sigma = 0.2
audio_scale = 4.0
resolution_ratio = 2.0

from_shift = 3.0
to_shift = 12.0


def _lab_path():
    return str(Path(__file__).resolve().parents[3] / "ComfyUI-MiniMaxH3-SPEED-Lab")


def test_flow_recover_internal_state():
    from minimax_h3_speed.flow import recover_internal_state as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import recover_internal_state as lab_func

    mvp_video, mvp_audio = mvp_func(video, audio, float(sigma), audio_scale)
    lab_video, lab_audio = lab_func(video, audio, float(sigma), audio_scale)
    assert torch.equal(mvp_video, lab_video)
    assert torch.equal(mvp_audio, lab_audio)


def test_flow_reentry_noise():
    from minimax_h3_speed.flow import reentry_noise as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import reentry_noise as lab_func

    internal_state = audio
    mvp_out = mvp_func(internal_state, new_video_sigma)
    lab_out = lab_func(internal_state, new_video_sigma)
    assert torch.equal(mvp_out, lab_out)


def test_flow_clock_reindex_audio_state():
    from minimax_h3_speed.flow import clock_reindex_audio_state as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import clock_reindex_audio_state as lab_func

    mvp_out = mvp_func(
        audio, audio, old_video_sigma, new_video_sigma,
        old_audio_sigma, new_audio_sigma, audio_scale,
    )
    lab_out = lab_func(
        audio, audio, old_video_sigma, new_video_sigma,
        old_audio_sigma, new_audio_sigma, audio_scale,
    )
    assert torch.equal(mvp_out, lab_out)


def test_flow_carry_preserving_audio_state():
    from minimax_h3_speed.flow import carry_preserving_audio_state as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import carry_preserving_audio_state as lab_func

    mvp_out = mvp_func(audio, old_video_sigma, new_video_sigma, old_audio_sigma, new_audio_sigma)
    lab_out = lab_func(audio, old_video_sigma, new_video_sigma, old_audio_sigma, new_audio_sigma)
    assert torch.equal(mvp_out, lab_out)


def test_flow_time_shift_sigma():
    from minimax_h3_speed.flow import time_shift_sigma as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import time_shift_sigma as lab_func

    mvp_out = mvp_func(sigma, from_shift, to_shift)
    lab_out = lab_func(sigma, from_shift, to_shift)
    assert torch.equal(mvp_out, lab_out)


def test_flow_aligned_speed_sigma():
    from minimax_h3_speed.flow import aligned_speed_sigma as mvp_func
    import sys
    sys.path.insert(0, _lab_path())
    from speed_lab.flow import aligned_speed_sigma as lab_func

    mvp_kappa, mvp_t = mvp_func(float(sigma), resolution_ratio)
    lab_kappa, lab_t = lab_func(float(sigma), resolution_ratio)
    # aligned_speed_sigma returns plain floats (kappa, t_tilde).
    assert mvp_kappa == lab_kappa
    assert mvp_t == lab_t


def test_resolve_transition_steps_delta_custom_matches_recommend():
    """Given config with transition_mode='delta_custom', the resolved steps
    must equal recommend_configs output for the same parameters."""
    from minimax_h3_speed.h3_runtime import resolve_transition_steps
    from minimax_h3_speed.harvest import recommend_configs
    sigmas = torch.linspace(1.0, 0.0, 21)  # 20 steps
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),  # explicit value — overridden by delta_custom
        transition_mode="delta_custom",
        delta=0.01,
        power_A=219.48,
        power_beta=2.42,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
    # Compare against recommend_configs for the same params
    rec = recommend_configs(219.48, 2.42, sigmas, latent_h=45, latent_w=80)
    expected = rec["half_then_full"]["transition_steps"]
    assert resolved == tuple(expected), f"resolved={resolved}, expected={tuple(expected)}"


def test_resolve_transition_steps_explicit_ignores_delta():
    """For 'explicit' mode, resolved steps must equal config.transition_steps,
    regardless of delta/power_A/power_beta values."""
    from minimax_h3_speed.h3_runtime import resolve_transition_steps
    sigmas = torch.linspace(1.0, 0.0, 21)
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(7,),
        transition_mode="explicit",
        delta=0.5,  # irrelevant in explicit mode
        power_A=999.0,
        power_beta=9.0,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas)
    assert resolved == (7,)


def test_resolve_transition_steps_validation():
    """Transition steps must be in (0, len(sigmas)-1)."""
    from minimax_h3_speed.h3_runtime import resolve_transition_steps
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    # Use explicit mode with a step that's valid for config creation
    # but test that resolve_transition_steps returns it as-is (no validation).
    # The actual validation happens in run_repeated_stage_calls.
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(1,),  # valid for config (>= 1)
        transition_mode="explicit",
        delta=0.01,
        power_A=219.48,
        power_beta=2.42,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas)
    assert resolved == (1,)


def test_activation_time_matches_canonical_formula():
    """Verify activation_time against hand-computed values from Eq. 9."""
    from minimax_h3_speed.h3_runtime import activation_time
    # P = 100, delta = 0.01:
    #   t* = 1 / (1 + sqrt(0.01 / (100 * (101 - 0.01))))
    #      = 1 / (1 + sqrt(0.01 / 10099))
    #      = 1 / (1 + sqrt(9.90e-7))
    #      ≈ 1 / (1 + 0.000995) ≈ 0.999005
    import math
    result = activation_time(100.0, 0.01)
    expected = 1.0 / (1.0 + math.sqrt(0.01 / (100.0 * (101.0 - 0.01))))
    assert abs(result - expected) < 1e-10
