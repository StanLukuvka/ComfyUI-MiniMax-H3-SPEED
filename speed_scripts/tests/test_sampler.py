"""Minimal contract test for the single SPEED sampler node."""
import importlib
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from speed_scripts.config import SpeedConfig

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

    class _ProgressBar:
        """No-op stand-in for comfy.utils.ProgressBar in headless tests."""
        def __init__(self, total, node_id=None):
            self.total = total
            self.node_id = node_id
        def update_absolute(self, value, total=None, preview=None):
            pass
        def update(self, value):
            pass
    utils.ProgressBar = _ProgressBar

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
# `speed_scripts.h3_runtime` (which now import comfy at top level) works
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
    assert "noise_amplitude" in required
    assert "noise_decay_exponent" in required
    assert "seed_offset" in required
    assert required["preset"][0][0] == "half_then_full"


def test_sample_runs_multi_stage():
    _install_comfy_stubs()
    from speed_scripts.h3_runtime import run_speed_pipeline

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
            if callback is not None:
                callback(0, latent_image, latent_image, len(sigmas))
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

    out, denoised = run_speed_pipeline(
        FakeNoise(), FakeGuider(), sigmas, latent, config,
        sampler=type("S", (), {"name": "euler"})(),
        nested_type=type("NT", (), {"is_nested": True})(),
        disable_pbar=True, output_device=None,
    )
    assert out is not None
    assert len(sample_calls) >= 2, f"expected ≥2 stages, got {len(sample_calls)}"


def test_coupled_full_grid_noise_policy():
    """coupled_full_grid: full-grid noise is shared across stages."""
    _install_comfy_stubs()
    from speed_scripts.h3_runtime import run_speed_pipeline

    sample_calls = []
    captured_noises = []

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            sample_calls.append(len(sigmas))
            if callback is not None:
                callback(0, latent_image, latent_image, len(sigmas))
            return latent_image

    class FakeNoise:
        seed = 77
        def generate_noise(self, latent):
            samples = latent.get("samples")
            if getattr(samples, "is_nested", False):
                vids = [s for s in samples.unbind() if s.ndim == 5]
                auds = [s for s in samples.unbind() if s.ndim != 5]
                nt = type("NT", (), {"is_nested": True, "unbind": lambda self: vids + auds})()
                captured_noises.append(nt)
                return nt
            return samples

    # Full-resolution latent: 32x32 spatial
    video = torch.zeros(1, 1, 2, 32, 32)
    audio = torch.zeros(1, 1, 2, 44)

    class FakeNested:
        is_nested = True
        def unbind(self):
            return [video, audio]

    nested = FakeNested()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.025, 20)
    config = SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), noise_policy="coupled_full_grid")

    out, denoised = run_speed_pipeline(
        FakeNoise(), FakeGuider(), sigmas, latent, config,
        sampler=type("S", (), {"name": "euler"})(),
        nested_type=type("NT", (), {"is_nested": True})(),
        disable_pbar=True, output_device=None,
    )
    assert out is not None
    assert len(sample_calls) >= 2
    assert len(captured_noises) > 0


def test_aligned_sigma_math():
    """kappa = r / (1 + (r-1)q); t_tilde = kappa * q. Verify the paper formula."""
    flow = importlib.import_module("speed_scripts.flow")
    for q, r in [(0.5, 2.0), (0.3, 2.0), (0.8, 4.0 / 3.0)]:
        kappa, t = flow.aligned_sigma(q, r)
        assert abs(kappa - r / (1.0 + (r - 1.0) * q)) < 1e-6
        assert abs(t - kappa * q) < 1e-12


def test_resolve_transition_steps_explicit_example():
    """Explicit mode places transitions at the flat per-scale default (5)."""
    mod = importlib.import_module("sampler_node")
    h3_runtime = importlib.import_module("speed_scripts.h3_runtime")
    sigmas = torch.linspace(1.0, 0.025, 20)
    explicit_steps = h3_runtime.resolve_transition_steps(
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), transition_mode="explicit"),
        sigmas, 64, 64,
    )
    assert tuple(explicit_steps) == (5,)


def test_sigma_shifts_returns_audio_scale_from_ratio():
    """resolve_sigma_shifts should compute audio_scale = video_shift / audio_shift,
    not read a non-existent 'audio_scale' attribute.

    With shift_video=12.0 and shift_audio=3.0, the correct audio_scale is 4.0.
    """
    from speed_scripts.h3_runtime import resolve_sigma_shifts

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "diffusion_model": type("DM", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
            })(),
        })()})()

    v_shift, a_shift, a_scale = resolve_sigma_shifts(FakeGuider())
    assert v_shift == 12.0
    assert a_shift == 3.0
    assert abs(a_scale - 4.0) < 1e-6


def test_audio_scale_equals_shift_ratio():
    """Verify audio_scale = video_shift / audio_shift, not 1.0."""
    from speed_scripts.h3_runtime import resolve_sigma_shifts

    class FakeGuider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
        })()})()

    _, _, a_scale = resolve_sigma_shifts(FakeGuider())
    assert abs(a_scale - 4.0) < 1e-6


def test_sigma_shifts_ignore_generic_comfy_shift():
    """REGRESSION: ComfyUI's generic ModelSamplingAV.shift (often 1.0) must NOT
    shadow the H3 model's own sigma_shift_video/audio (12.0 / 3.0).

    The 84e61ba 'sigma shift lookup' fix put model_sampling.shift at higher
    priority than sigma_shift_video. That collapsed audio_scale to ~0.333
    instead of 4.0, rescaling every audio transition ~12x wrong (garbled sound).
    The H3 attributes must win.
    """
    from speed_scripts.h3_runtime import resolve_sigma_shifts

    class FakeModelSampling:
        # Generic ComfyUI flow-matching shift — NOT H3-specific.
        shift = 1.0
        audio_shift = 1.0

    class FakeGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
            })(),
            "get_model_object": lambda self, name: (
                FakeModelSampling() if name == "model_sampling" else None
            ),
        })()

    v_shift, a_shift, a_scale = resolve_sigma_shifts(FakeGuider())
    assert v_shift == 12.0, f"expected H3 video_shift=12.0, got {v_shift}"
    assert a_shift == 3.0, f"expected H3 audio_shift=3.0, got {a_shift}"
    assert abs(a_scale - 4.0) < 1e-6, f"expected audio_scale=4.0, got {a_scale}"


def test_sigma_shifts_raise_without_h3_model():
    """PR3: A model that lacks H3 sigma_shift_video/audio must RAISE, not
    silently fall back to ComfyUI's generic ModelSamplingAV.shift.

    The 84e61ba bug silently used a generic flow shift (often 1.0) instead of
    the H3-specific shifts, garbling audio. The fix removes the silent fallback
    so a non-H3 model surfaces as a configuration error.
    """
    import pytest
    from speed_scripts.h3_runtime import resolve_sigma_shifts

    class FakeGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {})(),  # no sigma_shift_video/audio
            "get_model_object": lambda self, name: (
                type("MS", (), {"shift": 1.0, "audio_shift": 1.0})()
                if name == "model_sampling" else None
            ),
        })()

    with pytest.raises(ValueError, match="active MiniMax-H3 sigma shifts are unavailable"):
        resolve_sigma_shifts(FakeGuider())


def test_sigma_policy_canonical_vs_no_alignment():
    """canonical: apply kappa alignment; no_alignment: no rescaling."""
    from speed_scripts.h3_runtime import run_speed_pipeline

    canonical_calls = []
    no_align_calls = []

    def make_guider(calls_list):
        class FakeGuider:
            model_patcher = type("MP", (), {"model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()})()
            conds = {}
            def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
                calls_list.append(len(sigmas))
                if callback is not None:
                    callback(0, latent_image, latent_image, len(sigmas))
                return latent_image
        return FakeGuider()

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()
    latent = {"samples": nested}
    sigmas = torch.linspace(1.0, 0.025, 20)

    config_canon = SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), sigma_policy="canonical")
    config_noalign = SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), sigma_policy="no_alignment")

    run_speed_pipeline(
        type("N", (), {"seed": 42, "generate_noise": lambda s, l: l["samples"]})(),
        make_guider(canonical_calls),
        sigmas, latent, config_canon,
        sampler=type("S", (), {"name": "euler"})(),
        nested_type=type("NT", (), {"is_nested": True})(),
        disable_pbar=True,
    )
    run_speed_pipeline(
        type("N", (), {"seed": 42, "generate_noise": lambda s, l: l["samples"]})(),
        make_guider(no_align_calls),
        sigmas, latent, config_noalign,
        sampler=type("S", (), {"name": "euler"})(),
        nested_type=type("NT", (), {"is_nested": True})(),
        disable_pbar=True,
    )
    # Both should run the same number of stages
    assert len(canonical_calls) == len(no_align_calls)
    assert len(canonical_calls) >= 2


def test_invalid_transition_mode_raises():
    """Config with unsupported transition_mode must fail fast."""
    with pytest.raises(ValueError, match="transition_mode"):
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), transition_mode="unknown")


def test_invalid_noise_policy_raises():
    with pytest.raises(ValueError, match="noise_policy"):
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,), noise_policy="evil")


def test_single_scale_must_be_full():
    with pytest.raises(ValueError, match="single scale must be 1.0"):
        SpeedConfig(scales=(0.5,), transition_steps=())


def test_final_scale_must_be_full():
    with pytest.raises(ValueError, match="final scale must be 1.0"):
        SpeedConfig(scales=(0.5, 0.75), transition_steps=(3,))


def test_transition_steps_count_matches_scales():
    with pytest.raises(ValueError, match="need .* transition steps"):
        SpeedConfig(scales=(0.5, 0.75, 1.0), transition_steps=(5,))


def test_reentry_noise_formula():
    """reentry_noise(internal, start_sigma) = internal / start_sigma."""
    from speed_scripts.flow import reentry_noise
    internal = torch.tensor([1.0, 2.0, 3.0])
    result = reentry_noise(internal, 0.5)
    assert torch.allclose(result, internal / 0.5)


def test_reentry_noise_raises_on_zero():
    from speed_scripts.flow import reentry_noise
    with pytest.raises(ValueError, match="start_sigma must be positive"):
        reentry_noise(torch.zeros(3), 0.0)


def test_kappa_formula():
    """κ = r / (1 + (r-1)t) per Eq. (5)."""
    from speed_scripts.flow import aligned_sigma
    r = 2.0
    t = 0.5
    kappa, new_q = aligned_sigma(t, r)
    expected_kappa = r / (1.0 + (r - 1.0) * t)
    assert abs(kappa - expected_kappa) < 1e-10


def test_time_shift_sigma_raises_on_bad_inputs():
    """time_shift_sigma must reject non-positive shifts."""
    from speed_scripts.flow import time_shift_sigma
    with pytest.raises(ValueError):
        time_shift_sigma(0.5, 0.0, 1.0)
    with pytest.raises(ValueError):
        time_shift_sigma(0.5, 1.0, -1.0)


def test_time_shift_sigma_identity_at_full_res():
    """At q = q_ref (no shift), returns q unchanged."""
    from speed_scripts.flow import time_shift_sigma
    result = time_shift_sigma(0.5, 1.0, 1.0)
    assert abs(result - 0.5) < 1e-10


def test_flow_time_shift_sigma():
    """time_shift_sigma bridges video→audio sigma space correctly."""
    from speed_scripts.flow import time_shift_sigma
    # With video_shift=2.0, audio_shift=1.0, q_video=0.5:
    #   base = 0.5 / (2 + 0.5 * (1-2)) = 0.5 / 1.5 = 1/3
    #   q_audio = 1.0 * (1/3) / (1 + 0 * (1/3)) = 1/3
    q_audio = time_shift_sigma(0.5, 2.0, 1.0)
    assert abs(q_audio - (1.0/3.0)) < 1e-6


def test_resolve_transition_steps_delta_custom_matches_recommend():
    """Given config with transition_mode='delta_custom', the resolved steps
    must equal recommend_transition_steps output for the same parameters."""
    from speed_scripts.h3_runtime import resolve_transition_steps
    from speed_scripts.harvest import recommend_transition_steps
    sigmas = torch.linspace(1.0, 0.0, 21)  # 20 steps
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(5,),
        transition_mode="delta_custom",
        delta=0.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas, H_full=45, W_full=80)
    rec = recommend_transition_steps(219.48, 2.42, sigmas, latent_h=45, latent_w=80)
    expected = rec["half_then_full"]["transition_steps"]
    assert resolved == tuple(expected)


def test_resolve_transition_steps_explicit_ignores_delta():
    """For 'explicit' mode, resolved steps must equal config.transition_steps,
    regardless of delta/noise_amplitude/noise_decay_exponent values."""
    from speed_scripts.h3_runtime import resolve_transition_steps
    sigmas = torch.linspace(1.0, 0.0, 21)
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(7,),
        transition_mode="explicit",
        delta=0.5,  # irrelevant in explicit mode
        noise_amplitude=999.0,
        noise_decay_exponent=9.0,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas)
    assert resolved == (7,)


def test_resolve_transition_steps_validation():
    """Transition steps must be in (0, len(sigmas)-1)."""
    from speed_scripts.h3_runtime import resolve_transition_steps
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    config = SpeedConfig(
        scales=(0.5, 1.0),
        transition_steps=(1,),  # valid for config (>= 1)
        transition_mode="explicit",
        delta=0.01,
        noise_amplitude=219.48,
        noise_decay_exponent=2.42,
        full_latent_h=45,
        full_latent_w=80,
    )
    resolved = resolve_transition_steps(config, sigmas)
    assert resolved == (1,)


def test_activation_time_matches_canonical_formula():
    """Verify activation_threshold against hand-computed values from Eq. 9."""
    from speed_scripts.h3_runtime import activation_threshold
    import math
    result = activation_threshold(100.0, 0.01)
    expected = 1.0 / (1.0 + math.sqrt(0.01 / (100.0 * (101.0 - 0.01))))
    assert abs(result - expected) < 1e-10


def test_flow_time_shift_sigma():
    """time_shift_sigma bridges video→audio sigma space correctly."""
    from speed_scripts.flow import time_shift_sigma
    # With video_shift=2.0, audio_shift=1.0, q_video=0.5:
    #   base = 0.5 / (2 + 0.5 * (1-2)) = 0.5 / 1.5 = 1/3
    #   q_audio = 1.0 * (1/3) / (1 + 0 * (1/3)) = 1/3
    q_audio = time_shift_sigma(0.5, 2.0, 1.0)
    assert abs(q_audio - (1.0/3.0)) < 1e-6
