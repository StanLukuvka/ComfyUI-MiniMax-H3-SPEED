"""Contract and integration tests for the LatentClass-edition sampler nodes.

Tests the `_test` suffix variants (`MiniMaxH3SPEEDSamplerClassTest`,
`MiniMaxH3SPEEDSamplerManualClassTest`, `MiniMaxH3HarvestToConfigClassTest`)
imported from `nodes/class/`. These are structural mirrors of the live
`Class` nodes — the test surface is INPUT_TYPES, RETURN_TYPES, and
run-through with a fake guider/noise/sigmas stack.

Run with: pytest speed_scripts/tests/test_sampler_class.py -v
"""

from __future__ import annotations

import importlib

import pytest
import torch

from conftest import install_comfy_stubs

install_comfy_stubs()


# ---------------------------------------------------------------------------
# Helpers shared across all three node types
# ---------------------------------------------------------------------------

def make_fake_nested(video_t, audio_t=None):
    """Duck-typed NestedTensor stand-in used by h3_runtime.unpack_latent."""
    class _NT:
        is_nested = True
        def __init__(self, v, a):
            self._v = v
            self._a = a if a is not None else torch.zeros(1, 1, 2, 44)
        def unbind(self):
            return [self._v, self._a]
    return _NT(video_t, audio_t)


def make_fake_guider(return_latent=None):
    """Fake guider that returns `return_latent` unchanged from sample()."""
    class _FG:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def __init__(self, ret):
            self._ret = ret

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            if callback is not None:
                try:
                    callback(0, latent_image, latent_image, len(sigmas))
                except Exception:
                    pass
            return self._ret
    return _FG(return_latent)


def make_fake_noise(seed=42):
    """Fake noise object with generate_noise()."""
    class _FN:
        def __init__(self, seed=42):
            self.seed = seed
        def generate_noise(self, latent):
            samples = latent.get("samples")
            if getattr(samples, "is_nested", False):
                vids = [s for s in samples.unbind() if s.ndim == 5]
                auds = [s for s in samples.unbind() if s.ndim != 5]
                return type("NT", (), {"is_nested": True,
                                       "unbind": lambda self, v=vids, a=auds: v + a})()
            return samples
    return _FN(seed=seed)


# ---------------------------------------------------------------------------
# Automatic sampler — MiniMaxH3SPEEDSamplerClassTest
# ---------------------------------------------------------------------------

def test_class_node_in_node_class_mappings():
    """The _test variant must appear in its own NODE_CLASS_MAPPINGS."""
    install_comfy_stubs()
    mod = importlib.import_module("sampler_node_class_test")
    assert "MiniMaxH3SPEEDSamplerClassTest" in mod.NODE_CLASS_MAPPINGS
    assert mod.NODE_CLASS_MAPPINGS["MiniMaxH3SPEEDSamplerClassTest"] is mod.MiniMaxH3SPEEDSamplerClassTest


def test_class_node_return_types_and_function():
    """Automatic sampler returns LATENT x2 and uses 'sample' function."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert cls.RETURN_NAMES == ("output", "denoised_output")
    assert cls.FUNCTION == "sample"


def test_class_node_input_types_has_all_required():
    """INPUT_TYPES covers every required field the sampler needs."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    inputs = cls.INPUT_TYPES()
    required = inputs["required"]
    for key in ("noise", "guider", "sigmas", "latent_image", "stages",
                "noise_policy", "Tolerance (Delta)", "noise_amplitude",
                "noise_decay_exponent", "seed_offset"):
        assert key in required, f"missing required input: {key}"


def test_class_node_stages_default_3():
    """stages widget defaults to 3."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    inputs = cls.INPUT_TYPES()
    stages_widget = inputs["required"]["stages"]
    assert stages_widget[0] == "INT"
    assert stages_widget[1]["default"] == 3
    assert stages_widget[1]["min"] == 2
    assert stages_widget[1]["max"] == 4


def test_class_node_noise_policy_options():
    """noise_policy offers both 'direct_coarse' and 'coupled_full_grid'."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    inputs = cls.INPUT_TYPES()
    policy_widget = inputs["required"]["noise_policy"]
    assert policy_widget[0] == ["direct_coarse", "coupled_full_grid"]


def test_class_node_sample_runs_two_stages():
    """Calling sample() with stages=2 fires exactly two guider.sample() calls."""
    install_comfy_stubs()
    from speed_scripts.latent_class import LatentClass

    # Build the nested latent
    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}

    # Fake guider tracks call count
    call_count = [0]
    return_latent = make_fake_nested(
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 44),
    )

    class _CountingGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            call_count.__setitem__(0, call_count[0] + 1)
            if callback is not None:
                try:
                    callback(0, latent_image, latent_image, len(sigmas))
                except Exception:
                    pass
            return return_latent

    LatentClass.clear()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    out, denoised = cls().sample(
        noise=make_fake_noise(),
        guider=_CountingGuider(),
        sigmas=sigmas,
        latent_image=latent,
        stages=2,
        noise_policy="direct_coarse",
        Tolerance__Delta=0.01,
        noise_amplitude=7.394,
        noise_decay_exponent=0.62,
        seed_offset=10000,
    )
    assert out is not None
    assert denoised is not None
    assert call_count[0] >= 2, f"expected ≥2 stages, got {call_count[0]}"


def test_class_node_delta_alias_accepted():
    """The node accepts 'Delta', 'delta', 'tolerance' as aliases for Tolerance (Delta)."""
    install_comfy_stubs()
    from speed_scripts.latent_class import LatentClass

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}
    return_latent = make_fake_nested(
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 44),
    )

    class _Guider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            if callback is not None:
                try:
                    callback(0, latent_image, latent_image, len(sigmas))
                except Exception:
                    pass
            return return_latent

    LatentClass.clear()
    cls = importlib.import_module("sampler_node_class_test").MiniMaxH3SPEEDSamplerClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    # All aliases must not raise
    for alias in ("Tolerance (Delta)", "Tolerance", "tolerance", "delta", "Delta"):
        LatentClass.clear()
        try:
            cls().sample(
                noise=make_fake_noise(),
                guider=_Guider(),
                sigmas=sigmas,
                latent_image=latent,
                stages=2,
                **{alias: 0.01},
                noise_policy="direct_coarse",
                noise_amplitude=7.394,
                noise_decay_exponent=0.62,
                seed_offset=10000,
            )
        except TypeError as exc:
            if alias not in str(exc):
                raise
            pytest.fail(f"alias '{alias}' not accepted: {exc}")


# ---------------------------------------------------------------------------
# Manual sampler — MiniMaxH3SPEEDSamplerManualClassTest
# ---------------------------------------------------------------------------

def test_manual_class_node_in_mappings():
    """Manual sampler _test variant is in its own NODE_CLASS_MAPPINGS."""
    install_comfy_stubs()
    mod = importlib.import_module("sampler_node_manual_class_test")
    assert "MiniMaxH3SPEEDSamplerManualClassTest" in mod.NODE_CLASS_MAPPINGS


def test_manual_class_node_return_types():
    """Manual sampler returns LATENT x2 and uses 'sample' function."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_manual_class_test").MiniMaxH3SPEEDSamplerManualClassTest
    assert cls.RETURN_TYPES == ("LATENT", "LATENT")
    assert cls.RETURN_NAMES == ("output", "denoised_output")
    assert cls.FUNCTION == "sample"


def test_manual_class_node_has_transition_widgets():
    """INPUT_TYPES exposes all four (goal, resolution) widget pairs."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_node_manual_class_test").MiniMaxH3SPEEDSamplerManualClassTest
    inputs = cls.INPUT_TYPES()
    required = inputs["required"]
    for n in range(1, 5):
        assert f"transition_goal_{n}" in required
        assert f"transition_resolution_{n}" in required
    assert "ratio_mode" in required


def test_manual_class_node_sample_runs_two_stages():
    """Manual sampler fires exactly two stages when only goal_1/resolution_1 are set."""
    install_comfy_stubs()
    from speed_scripts.latent_class import LatentClass

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}
    return_latent = make_fake_nested(
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 44),
    )

    call_count = [0]

    class _CountingGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            call_count.__setitem__(0, call_count[0] + 1)
            if callback is not None:
                try:
                    callback(0, latent_image, latent_image, len(sigmas))
                except Exception:
                    pass
            return return_latent

    LatentClass.clear()
    cls = importlib.import_module("sampler_node_manual_class_test").MiniMaxH3SPEEDSamplerManualClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    out, denoised = cls().sample(
        noise=make_fake_noise(),
        guider=_CountingGuider(),
        sigmas=sigmas,
        latent_image=latent,
        noise_policy="direct_coarse",
        seed_offset=10000,
        ratio_mode="steps",
        transition_goal_1=5,
        transition_resolution_1=0.5,
        transition_goal_2=0,   # disabled
        transition_resolution_2=0,
        transition_goal_3=0,
        transition_resolution_3=0,
        transition_goal_4=15,  # final stage at full res
        transition_resolution_4=1.0,
    )
    assert out is not None
    assert call_count[0] >= 2


def test_manual_class_node_ratio_mode():
    """ratio_mode='ratio' accepts fractional goals (0 < goal <= 1)."""
    install_comfy_stubs()
    from speed_scripts.latent_class import LatentClass

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}
    return_latent = make_fake_nested(
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 44),
    )

    class _Guider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            if callback is not None:
                try:
                    callback(0, latent_image, latent_image, len(sigmas))
                except Exception:
                    pass
            return return_latent

    LatentClass.clear()
    cls = importlib.import_module("sampler_node_manual_class_test").MiniMaxH3SPEEDSamplerManualClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    # ratio mode with goal=0.25 -> step boundary at round(0.25*19)
    out, denoised = cls().sample(
        noise=make_fake_noise(),
        guider=_Guider(),
        sigmas=sigmas,
        latent_image=latent,
        noise_policy="direct_coarse",
        seed_offset=10000,
        ratio_mode="ratio",
        transition_goal_1=0.25,
        transition_resolution_1=0.5,
        transition_goal_2=0,
        transition_resolution_2=0,
        transition_goal_3=0,
        transition_resolution_3=0,
        transition_goal_4=1.0,  # final stage at full res
        transition_resolution_4=1.0,
    )
    assert out is not None


def test_manual_class_node_rejects_single_stage():
    """Manual sampler raises when fewer than two stages are active."""
    install_comfy_stubs()
    from speed_scripts.latent_class import LatentClass

    video = torch.zeros(1, 1, 2, 8, 8)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}
    return_latent = make_fake_nested(
        torch.zeros(1, 1, 2, 8, 8),
        torch.zeros(1, 1, 2, 44),
    )

    class _Guider:
        model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        conds = {}
        def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
            return return_latent

    LatentClass.clear()
    cls = importlib.import_module("sampler_node_manual_class_test").MiniMaxH3SPEEDSamplerManualClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    with pytest.raises(ValueError, match="at least two active stages"):
        cls().sample(
            noise=make_fake_noise(),
            guider=_Guider(),
            sigmas=sigmas,
            latent_image=latent,
            noise_policy="direct_coarse",
            seed_offset=10000,
            ratio_mode="steps",
            transition_goal_1=5,
            transition_resolution_1=0.5,
            transition_goal_2=0,  # stage 2 disabled -> only one active stage
            transition_resolution_2=0,
            transition_goal_3=0,
            transition_resolution_3=0,
            transition_goal_4=0,
            transition_resolution_4=0,
        )


# ---------------------------------------------------------------------------
# Harvest node — MiniMaxH3HarvestToConfigClassTest
# ---------------------------------------------------------------------------

def test_harvest_class_node_in_mappings():
    """Harvest _test variant is in its own NODE_CLASS_MAPPINGS."""
    install_comfy_stubs()
    mod = importlib.import_module("sampler_sigma_manual_class_test")
    assert "MiniMaxH3HarvestToConfigClassTest" in mod.NODE_CLASS_MAPPINGS


def test_harvest_class_node_return_types():
    """Harvest returns STRING and LATENT (calibration + diagnostic_latent)."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_sigma_manual_class_test").MiniMaxH3HarvestToConfigClassTest
    assert cls.RETURN_TYPES == ("STRING", "LATENT")
    assert cls.RETURN_NAMES == ("calibration", "diagnostic_latent")
    assert cls.FUNCTION == "harvest"


def test_harvest_class_node_input_types():
    """INPUT_TYPES has noise, guider, sigmas, latent_image + optional Tolerance."""
    install_comfy_stubs()
    cls = importlib.import_module("sampler_sigma_manual_class_test").MiniMaxH3HarvestToConfigClassTest
    inputs = cls.INPUT_TYPES()
    required = inputs["required"]
    for key in ("noise", "guider", "sigmas", "latent_image"):
        assert key in required
    assert "Tolerance (Delta)" in inputs.get("optional", {})


def test_harvest_class_node_runs_euler_and_emits_json():
    """harvest() fires the guider sampler and returns a valid calibration JSON."""
    install_comfy_stubs()

    video = torch.zeros(1, 1, 2, 64, 64)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}

    # The harvest node's own compute_video_residual needs a 5D tensor per frame.
    # Return a NestedTensor that produces a 5D tensor for the video stream.
    return_video = torch.randn(1, 1, 2, 64, 64)
    return_audio = torch.zeros(1, 1, 2, 44)

    class _NestedOut:
        is_nested = True
        def unbind(self):
            return [return_video, return_audio]

    callback_results = []

    class _HarvestGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            # Harvest fires callbacks per step with (step, denoised, x, total_steps)
            if callback is not None:
                for i in range(len(sigmas) - 1):
                    # Use non-zero tensors so residual is non-zero for spectral fit
                    x_step = torch.randn_like(return_video)
                    denoised_step = torch.randn_like(return_video)
                    try:
                        callback(i, denoised_step, x_step, len(sigmas))
                    except Exception:
                        pass
            return _NestedOut()

    cls = importlib.import_module("sampler_sigma_manual_class_test").MiniMaxH3HarvestToConfigClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    cal_json, out_latent = cls().harvest(
        noise=make_fake_noise(),
        guider=_HarvestGuider(),
        sigmas=sigmas,
        latent_image=latent,
        Tolerance__Delta=0.01,
    )

    import json
    cal = json.loads(cal_json)
    assert "noise_amplitude" in cal
    assert "noise_decay_exponent" in cal
    assert "delta" in cal
    assert "r2" in cal
    assert "health" in cal
    assert out_latent is not None


def test_harvest_class_node_returns_error_on_no_captures():
    """When the callback never fires, harvest returns an error JSON (not an exception)."""
    install_comfy_stubs()

    video = torch.zeros(1, 1, 2, 64, 64)
    audio = torch.zeros(1, 1, 2, 44)
    nested = make_fake_nested(video, audio)
    latent = {"samples": nested}

    class _NoCallbackGuider:
        model_patcher = type("MP", (), {
            "model": type("M", (), {
                "sigma_shift_video": 12.0,
                "sigma_shift_audio": 3.0,
                "process_latent_out": lambda s, x: x,
            })()
        })()
        conds = {}

        def sample(self, noise, latent_image, sampler, sigmas,
                   callback=None, disable_pbar=True, seed=0, **kwargs):
            # No callback fires — harvest must handle this gracefully
            return make_fake_nested(video, audio)

    cls = importlib.import_module("sampler_sigma_manual_class_test").MiniMaxH3HarvestToConfigClassTest
    sigmas = torch.linspace(1.0, 0.025, 20)

    cal_json, out_latent = cls().harvest(
        noise=make_fake_noise(),
        guider=_NoCallbackGuider(),
        sigmas=sigmas,
        latent_image=latent,
    )

    import json
    cal = json.loads(cal_json)
    assert "error" in cal or "no_captures" in cal.get("error", "")
