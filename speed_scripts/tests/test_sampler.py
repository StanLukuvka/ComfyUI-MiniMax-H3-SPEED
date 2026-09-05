"""Minimal contract test for the single SPEED sampler node."""
import importlib
import math

import pytest
import torch
from speed_scripts.config import SpeedConfig

# One canonical comfy stub installer (see conftest) — imported under the old
# name so existing call sites stay valid.
from conftest import (
    install_comfy_stubs as _install_comfy_stubs,
    make_fake_guider,
    make_fake_noise,
    make_nested,
)


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
                "stages",):
        assert key in required, f"missing required input: {key}"
    # automatic node is delta_custom only — no preset/transition_mode (explicit lives on Manual)
    assert "preset" not in required
    assert "transition_mode" not in required
    # delta_custom path is enabled with sigma-harvest calibration
    assert "Tolerance (Delta)" in required or "delta" in required
    assert "noise_amplitude" in required
    assert "noise_decay_exponent" in required
    assert "seed_offset" in required
    assert required["stages"][0] == "INT"
    assert required["stages"][1]["default"] == 3


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
        disable_pbar=True,
    )
    run_speed_pipeline(
        type("N", (), {"seed": 42, "generate_noise": lambda s, l: l["samples"]})(),
        make_guider(no_align_calls),
        sigmas, latent, config_noalign,
        sampler=type("S", (), {"name": "euler"})(),
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


def _install_preview_fakes():
    """Install the comfy stub modules needed by `_build_preview_callback` and
    `_wrap_preview_callback` so the runtime exercises the stock
    `latent_preview.prepare_callback` path end-to-end."""
    import sys
    from types import ModuleType

    _install_comfy_stubs()

    # Give the fake model the attributes `get_previewer` needs so it returns
    # a real previewer (or None gracefully — that's also valid and tested).
    model = sys.modules["comfy"].model
    model.load_device = type("D", (), {"device": "cpu"})()
    model.model.latent_format = type("LF", (), {})()


def _preview_fake_guider(sample_calls=None):
    class FakeGuider:
        class _Model:
            sigma_shift_video = 12.0
            sigma_shift_audio = 3.0
            # ComfyUI stores latent_rgb_factors on the model instance, not on
            # the latent_format class. When present, `get_previewer` finds a
            # previewer and `prepare_callback` produces preview bytes each step.
            # When absent, the callback still runs and updates the bar — just
            # without a JPEG image.
            latent_rgb_factors = [[0.1] * 3 for _ in range(24)]
            latent_rgb_factors_bias = [0.0, 0.0, 0.0]

            def process_latent_out(self, x):
                return x

        model_patcher = type("MP", (), {"model": _Model()})()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            if sample_calls is not None:
                sample_calls.append(len(sigmas))
            # Fire one callback per denoise step, like a real sampler.
            for i in range(len(sigmas) - 1):
                callback(i, latent_image, latent_image, len(sigmas) - 1)
            return latent_image

    return FakeGuider()


def _preview_fake_noise():
    class FakeNoise:
        seed = 7

        def generate_noise(self, latent):
            samples = latent.get("samples")
            if getattr(samples, "is_nested", False):
                parts = list(samples.unbind())
                return type("NT", (), {"is_nested": True, "unbind": lambda s=parts: list(s)})()
            return samples

    return FakeNoise()


def _preview_fake_latent():
    # H3 video stream shape: (B=1, C=24, T=2, H=8, W=8) — must match the
    # 24-channel factor matrix the preview callback expects.
    video = torch.zeros(1, 24, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)

    class FakeNested:
        is_nested = True

        def unbind(self):
            return [video, audio]

    return {"samples": FakeNested()}


def test_preview_callback_global_steps():
    """Stock `latent_preview.prepare_callback` sees continuous global steps.

    The runtime builds the stock callback once for the full denoise total
    (matches `SamplerCustomAdvanced`), then wraps it per stage to remap the
    local `step` into a global timeline. The resulting `pbar.update_absolute`
    calls must run continuously 1..19 across stages instead of resetting.
    """
    import sys
    from types import ModuleType

    pbar_updates: list = []
    _install_comfy_stubs()
    # Install a stock-style `latent_preview.prepare_callback` that hands the
    # closure the test fakes need: it builds a real `ProgressBar` and writes
    # the shared x0 dict, exactly like `latent_preview.py:120-129`. The runtime
    # doesn't know or care that this is a test fake — it just forwards.
    utils = sys.modules["comfy.utils"]

    class FakePbar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, value, total=None, preview=None):
            pbar_updates.append((value, total if total is not None else self.total, preview))

    utils.ProgressBar = FakePbar

    def fake_prepare_callback(model_patcher, steps, x0_output_dict=None):
        pbar = FakePbar(steps)
        def _cb(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0
            pbar.update_absolute(step + 1, total_steps, None)
        return _cb

    preview_mod = ModuleType("comfy.latent_preview")
    preview_mod.prepare_callback = fake_prepare_callback
    sys.modules["comfy.latent_preview"] = preview_mod
    sys.modules["comfy"].latent_preview = preview_mod

    import importlib
    h3_runtime = importlib.import_module("speed_scripts.h3_runtime")
    importlib.reload(h3_runtime)

    sigmas = torch.linspace(1.0, 0.025, 20)
    x0_output: dict = {}
    out, denoised = h3_runtime.run_speed_pipeline(
        _preview_fake_noise(), _preview_fake_guider(), sigmas, _preview_fake_latent(),
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,)),
        sampler=object(), disable_pbar=False, output_device=None,
        x0_output=x0_output,
    )
    assert out is not None and denoised is not None

    # 19 denoise steps total. The bar must see values 1..19 exactly once each,
    # in order, with total=19 — proof that the stage offsets are remapped.
    values = [v for v, _t, _p in pbar_updates]
    assert values == list(range(1, 20)), (
        f"preview bar not continuous across stages: {values}"
    )
    assert all(t == 19 for _v, t, _p in pbar_updates)
    # The shared x0 dict is the same one the denoised block reads from.
    assert "x0" in x0_output, "stock callback must write the shared x0 dict"

    # Reset so subsequent tests start clean.
    sys.modules.pop("comfy.latent_preview", None)
    _install_comfy_stubs()
    importlib.reload(h3_runtime)


def test_fallback_pbar_when_no_preview():
    """Without `comfy.latent_preview`, a plain progress bar still runs globally."""
    import sys

    _install_comfy_stubs()
    # Drop `latent_preview` entirely so `_build_preview_callback` returns None
    # and the runtime's fallback path (plain `comfy.utils.ProgressBar`) fires.
    sys.modules.pop("comfy.latent_preview", None)
    utils = sys.modules["comfy.utils"]
    pbar_updates: list = []

    class FakePbar:
        def __init__(self, total):
            self.total = total

        def update_absolute(self, value, total=None, preview=None):
            pbar_updates.append((value, total if total is not None else self.total))

    utils.ProgressBar = FakePbar

    import importlib
    h3_runtime = importlib.import_module("speed_scripts.h3_runtime")
    importlib.reload(h3_runtime)

    sigmas = torch.linspace(1.0, 0.025, 20)
    h3_runtime.run_speed_pipeline(
        _preview_fake_noise(), _preview_fake_guider(), sigmas, _preview_fake_latent(),
        SpeedConfig(scales=(0.5, 1.0), transition_steps=(5,)),
        sampler=object(), disable_pbar=False, output_device=None,
    )
    values = sorted(v for v, _t in pbar_updates)
    assert values == list(range(1, 20)), f"fallback bar not continuous: {values}"

    _install_comfy_stubs()
    importlib.reload(h3_runtime)
