"""Node tests for the continuous SPEED Sigma Harvest (plan §55, §56).

Drives `run_speed_pipeline` through the node with the same fake
guider/noise infrastructure as the commit-2 observer tests: one SPEED
generation (no native pre-pass, no second run), three outputs, per-callback
records at stride 1, both measurement bases, strict-JSON output, and
Automatic-vs-harvest config equivalence.

The node builds its SpeedConfig through the shared automatic builder
(delta_custom), so expected stage boundaries are computed here from the
runtime's own public scheduling math (`power_at_frequency`,
`activation_threshold`, `first_step_below`) instead of hardcoding indices.
"""

import importlib
import importlib.util
import json
import math

import pytest
import torch

from conftest import install_comfy_stubs as _install_comfy_stubs

_install_comfy_stubs()

import speed_scripts.h3_runtime as h3_runtime

SIGMAS_10 = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

DEFAULTS = dict(delta=0.01, noise_amplitude=7.394, noise_decay_exponent=0.62)


class SpectralGuider:
    """Fake guider whose x0 payload carries deterministic spectral content.

    sample() echoes the latent and fires one callback per denoising
    interval with a non-zero pattern x0 (so the radial DCT fits have real
    bins) and the raw state as x — the shapes the runtime's
    `_wrap_observer_callback` hands to an observer.
    """

    def __init__(self):
        self.sigma_calls = []
        self.model_patcher = type("MP", (), {"model": type("M", (), {
            "sigma_shift_video": 12.0,
            "sigma_shift_audio": 3.0,
            "process_latent_out": lambda s, x: x,
        })()})()
        self.conds = {}

    def sample(self, noise, latent_image, sampler, sigmas, callback=None, **kwargs):
        self.sigma_calls.append([float(s) for s in sigmas])
        if callback is not None:
            n_steps = len(sigmas) - 1
            for i in range(n_steps):
                progress = (i + 1) / n_steps
                streams = []
                for stream in latent_image.unbind():
                    # Deterministic non-trivial pattern: a seeded random
                    # field has a dense DCT spectrum, so every radial bin
                    # is positive and the fits succeed at any resolution.
                    generator = torch.Generator().manual_seed(1234)
                    pattern = torch.rand(stream.shape, generator=generator) * 2.0 - 1.0
                    streams.append(pattern * progress)
                x0 = type("NT", (), {
                    "is_nested": True,
                    "unbind": lambda self, _s=streams: list(_s),
                })()
                callback(i, x0, latent_image, n_steps)
        return latent_image


class EchoNoise:
    """Noise that regenerates the (video, audio) streams unchanged."""

    seed = 42

    def generate_noise(self, latent):
        samples = latent.get("samples")
        vids = [s for s in samples.unbind() if s.ndim == 5]
        auds = [s for s in samples.unbind() if s.ndim != 5]
        return type("NT", (), {"is_nested": True, "unbind": lambda self: vids + auds})()


def make_latent(t=2, h=8, w=8, **extra):
    video = torch.zeros(1, 1, t, h, w)
    audio = torch.zeros(1, 1, 2, 44)
    nested = type("NT", (), {"is_nested": True, "unbind": lambda self: [video, audio]})()
    latent = {"samples": nested}
    latent.update(extra)
    return latent


def expected_boundaries(sigmas, full_h, full_w, scales, delta, A, beta):
    """Resolve delta_custom boundaries with the runtime's own public math."""
    boundaries = []
    for scale in scales[:-1]:
        omega = scale * min(full_h, full_w) / 2.0
        p = h3_runtime.power_at_frequency(omega, A, beta)
        threshold = h3_runtime.activation_threshold(p, delta)
        boundaries.append(h3_runtime._find_first_step_below(sigmas, threshold))
    return tuple(boundaries)


def _expected(sigmas, full_h, full_w, scales=(0.25, 0.5, 0.75, 1.0)):
    return expected_boundaries(
        sigmas, full_h, full_w, scales,
        delta=DEFAULTS["delta"],
        A=DEFAULTS["noise_amplitude"],
        beta=DEFAULTS["noise_decay_exponent"],
    )


def _run_node(sigmas=SIGMAS_10, latent=None, measurement_mode="both",
              analysis_stride=1, **overrides):
    """Run the harvest node's sample() and return (document, info).

    run_speed_pipeline is forwarded to the real implementation (the node
    must drive the actual pipeline); call counting happens via the guider's
    sigma_calls.
    """
    mod = importlib.import_module("sampler_speed_sigma_harvest_node")
    node = mod.MiniMaxH3SPEEDSigmaHarvest()
    guider = SpectralGuider()
    latent = latent if latent is not None else make_latent()
    kwargs = dict(DEFAULTS)
    kwargs.update(overrides)
    result = node.sample(
        noise=EchoNoise(),
        guider=guider,
        sigmas=sigmas,
        latent_image=latent,
        stages=4,
        noise_policy="direct_coarse",
        measurement_mode=measurement_mode,
        analysis_stride=analysis_stride,
        **kwargs,
    )
    json_text, output, denoised = result
    return json.loads(json_text), {
        "output": output,
        "denoised": denoised,
        "guider": guider,
        "latent": latent,
        "module": mod,
    }


def test_node_schema_and_registration():
    """§48/§55: registration mapping, three named outputs, generation
    inputs identical to the Automatic sampler."""
    mod = importlib.import_module("sampler_speed_sigma_harvest_node")
    cls = mod.MiniMaxH3SPEEDSigmaHarvest
    assert cls.RETURN_TYPES == ("STRING", "LATENT", "LATENT")
    assert cls.RETURN_NAMES == ("harvest_json", "output", "denoised_output")
    assert cls.CATEGORY == "sampling/minimax_h3_speed/diagnostics"
    assert "MiniMaxH3SPEEDSigmaHarvest" in mod.NODE_CLASS_MAPPINGS

    # §48: the pack's registration loop picks the node up. Import the
    # package directory as a module by path (its name is not importable
    # as a plain identifier).
    from pathlib import Path

    pack_path = Path(__file__).resolve().parents[2] / "__init__.py"
    spec = importlib.util.spec_from_file_location("pack_init", pack_path)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    assert "sampler_speed_sigma_harvest_node" in pack._NODE_MODULES
    aggregated = {}
    for name in pack._NODE_MODULES:
        node_mod = importlib.import_module(name)
        aggregated.update(getattr(node_mod, "NODE_CLASS_MAPPINGS", {}))
    assert "MiniMaxH3SPEEDSigmaHarvest" in aggregated

    required = cls.INPUT_TYPES()["required"]
    for key in ("noise", "guider", "sigmas", "latent_image", "stages",
                "noise_policy", "Tolerance (Delta)", "noise_amplitude",
                "noise_decay_exponent", "seed_offset", "measurement_mode",
                "analysis_stride", "smoothing_alpha",
                "boundary_band_half_width", "store_radial_profiles"):
        assert key in required, f"missing required input: {key}"
    auto_required = importlib.import_module(
        "sampler_node"
    ).MiniMaxH3SPEEDSampler.INPUT_TYPES()["required"]
    for key, spec in auto_required.items():
        assert required[key] == spec, f"generation input {key} drifted from the Automatic sampler"


def test_single_speed_pass_three_outputs_metadata_preserved():
    """§55: exactly one SPEED generation (four stage guider.sample calls,
    no native pre-pass); outputs flow from that run and keep latent
    metadata."""
    latent = make_latent(metadata="keep", extra_key=[1, 2, 3])
    document, info = _run_node(latent=latent)

    assert len(info["guider"].sigma_calls) == 4
    assert document["records"], "no telemetry collected"

    assert info["output"] is not latent
    assert info["output"]["metadata"] == "keep"
    assert info["output"]["extra_key"] == [1, 2, 3]
    assert info["denoised"] is not latent
    assert info["denoised"]["metadata"] == "keep"
    assert info["denoised"]["extra_key"] == [1, 2, 3]


def test_records_every_callback_with_stage_metadata():
    """§55: stride=1 records every denoising interval; stage metadata and
    transition events match the delta_custom resolution the Automatic
    config produces for this latent."""
    latent = make_latent(t=2, h=72, w=80)
    document, info = _run_node(latent=latent)

    records = document["records"]
    assert [r["callback_index"] for r in records] == list(range(10))
    assert [r["global_schedule_index"] for r in records] == list(range(10))

    boundaries = _expected(SIGMAS_10, 72, 80)
    scales = (0.25, 0.5, 0.75, 1.0)
    stage_starts = (0,) + boundaries
    for record in records:
        expected_stage = max(
            i for i, start in enumerate(stage_starts)
            if record["global_schedule_index"] >= start
        )
        assert record["stage_index"] == expected_stage
        assert record["stage_scale"] == scales[expected_stage]

    transitions = document["transitions"]
    assert [t["transition_index"] for t in transitions] == [0, 1, 2]
    assert [t["global_schedule_index"] for t in transitions] == list(boundaries)


def test_both_bases_and_mode_respected():
    """§55: mode=both records both bases; x0_only omits the residual key;
    residual_only omits the x0 fit. Uses a full-size latent so the radial
    fits have enough bins to succeed (plan §21: at least 3 non-DC bins)."""
    latent = make_latent(t=2, h=72, w=80)
    document, _ = _run_node(latent=latent)
    record = document["records"][0]
    assert record["x0_signal"]["fit"]["status"] == "ok"
    assert record["residual"]["fit"]["status"] == "ok"
    assert document["measurement_bases"] == [
        "denoised_x0_video", "residual_x_minus_x0_video",
    ]

    x0_only, _ = _run_node(latent=latent, measurement_mode="x0_only")
    x0_record = x0_only["records"][0]
    assert "x0_signal" in x0_record
    assert "residual" not in x0_record
    assert x0_only["measurement_bases"] == ["denoised_x0_video"]

    residual_only, _ = _run_node(latent=latent, measurement_mode="residual_only")
    res_record = residual_only["records"][0]
    assert "residual" in res_record
    assert "x0_signal" not in res_record
    assert residual_only["measurement_bases"] == ["residual_x_minus_x0_video"]


def test_tiny_latent_fit_fails_explicitly():
    """§21/§43: a 2x2 coarse stage has a single non-DC radial bin, below
    the 3-bin minimum, so the fit fails explicitly — a JSON-safe
    fit_failed record with null numerics, not NaN and not a crash."""
    document, _ = _run_node()
    record = document["records"][0]
    fit = record["x0_signal"]["fit"]
    assert fit["status"] == "fit_failed"
    assert fit["A"] is None and fit["beta"] is None and fit["r_squared"] is None
    # The boundary block still records direct power (measurable with bins),
    # and the run completes with strict JSON.
    json.dumps(document, allow_nan=False)


def test_boundary_and_fit_predictions_present():
    """§27-§34: the boundary block carries direct power, thresholds,
    eligibility and the EMA; fit predictions label current vs extrapolated."""
    latent = make_latent(t=2, h=72, w=80)
    document, _ = _run_node(latent=latent)

    stage0 = next(
        r for r in document["records"] if r["stage_index"] == 0
    )["x0_signal"]
    boundary = stage0["current_boundary"]
    assert boundary["direct_available"] is True
    assert boundary["omega"] == pytest.approx(0.25 * 72 / 2)
    assert boundary["power_point"] > 0
    assert boundary["power_band_mean"] > 0
    assert 0.0 < boundary["activation_threshold_point"] < 1.0
    assert isinstance(boundary["eligible_point"], bool)
    assert boundary["ema_power"] > 0
    assert 0.0 < boundary["ema_threshold"] < 1.0

    predictions = stage0["fit_predictions"]
    assert predictions, "fit predictions missing"
    labels = [p["measurement"] for p in predictions]
    assert "power_law_fit" in labels
    assert "fit_extrapolation" in labels
    for p in predictions:
        assert isinstance(p["predicted_original_step"], int)

    # Final-stage records have no current boundary (no next transition).
    final_records = [r for r in document["records"] if r["stage_index"] == 3]
    assert final_records
    for r in final_records:
        assert "current_boundary" not in r["x0_signal"]


def test_coincident_transition_zero_step_stages_recorded():
    """§16/§53 through the node: at 8x8 the automatic calibration resolves
    to coincident boundaries [1, 1, 1]; the zero-step intermediate stages
    emit no records, both transitions still fire, and the final stage
    carries the remaining callbacks."""
    document, _ = _run_node()

    boundaries = _expected(SIGMAS_10, 8, 8)
    # The 8x8 calibration really does quantize to a coincident boundary,
    # otherwise this test would not exercise the zero-step path.
    assert len(set(boundaries)) < len(boundaries)

    transitions = document["transitions"]
    assert [t["global_schedule_index"] for t in transitions] == list(boundaries)
    assert [(t["from_stage"], t["to_stage"]) for t in transitions] == [(0, 1), (1, 2), (2, 3)]

    stage_indices = {r["stage_index"] for r in document["records"]}
    zero_step_stages = {
        t["to_stage"] for t in transitions
        if t["to_stage"] not in stage_indices
    }
    assert zero_step_stages, "expected at least one zero-step intermediate stage"


def test_strict_json_no_nan():
    """§55: the document round-trips a strict parser; no NaN anywhere."""
    document, _ = _run_node(measurement_mode="both", analysis_stride=1)
    text = json.dumps(document, allow_nan=False)  # must not raise
    reparsed = json.loads(text, parse_constant=_reject_constant)
    _no_nan(reparsed)


def _reject_constant(value):
    raise AssertionError(f"non-JSON constant in document: {value}")


def _no_nan(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for v in value.values():
            _no_nan(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _no_nan(v)


def test_ema_seeded_from_first_measurement_not_calibration():
    """§36: the stage's first EMA value equals its first raw measurement —
    never a power derived from the static A/beta."""
    document, _ = _run_node()
    stage_records = [r for r in document["records"] if r["stage_index"] == 0]
    first = stage_records[0]["x0_signal"]["current_boundary"]
    assert first["ema_power"] == pytest.approx(first["power_point"])


def test_stride_respects_analysis_stride():
    """§25: stride=2 analyses every second callback; unmeasured callbacks
    still get skeleton records."""
    document, _ = _run_node(analysis_stride=2)

    records = document["records"]
    assert len(records) == 10
    measured = [r for r in records if "x0_signal" in r]
    assert [r["callback_index"] for r in measured] == [0, 2, 4, 6, 8]
    skeleton = next(r for r in records if r["callback_index"] == 1)
    assert "x0_signal" not in skeleton and "residual" not in skeleton


def test_summary_is_scalar_per_stage():
    """§41: per-stage summary blocks exist with scalar fit statistics."""
    latent = make_latent(t=2, h=72, w=80)
    document, _ = _run_node(latent=latent)

    summary = document["summary"]
    assert "0" in summary
    stage0 = summary["0"]
    boundaries = _expected(SIGMAS_10, 72, 80)
    assert stage0["static_planned_transition_step"] == boundaries[0]
    for name in ("x0_beta", "x0_A", "residual_beta", "residual_A"):
        block = stage0[name]
        assert block["first"] is not None
        assert block["min"] <= block["mean"] <= block["max"]
    assert stage0["x0_r2_mean"] is not None


def test_top_level_schema_fields():
    """§39: schema_version, mode, measurement_bases, static_scheduler,
    analysis block."""
    document, _ = _run_node()

    assert document["schema_version"] == 1
    assert document["type"] == "minimax_h3_speed_sigma_harvest"
    assert document["mode"] == "observational"
    assert document["adaptive_control"] is False
    assert document["sampler"] == "euler"
    assert document["measurement_bases"] == [
        "denoised_x0_video", "residual_x_minus_x0_video",
    ]
    scheduler = document["static_scheduler"]
    assert scheduler["stages"] == 4
    assert scheduler["scales"] == [0.25, 0.5, 0.75, 1.0]
    assert scheduler["resolved_transition_steps"] == list(_expected(SIGMAS_10, 8, 8))
    assert scheduler["original_sigmas"] == SIGMAS_10.tolist()
    assert scheduler["delta"] == 0.01
    assert document["analysis"]["stride"] == 1
    assert document["analysis"]["measurement_mode"] == "both"


def test_automatic_vs_harvest_config_equivalence():
    """§56: identical inputs -> equal SpeedConfig fields on both node paths."""
    from speed_scripts.tests.test_automatic_config import _latent, _node_config

    kwargs = dict(
        stages=3,
        noise_policy="coupled_full_grid",
        delta=0.005,
        noise_amplitude=12.454,
        noise_decay_exponent=0.819,
        seed_offset=777,
    )
    auto_cfg = _node_config(_latent(45, 80), **kwargs)

    captured = {}

    def spy(noise, guider, sigmas, latent_image, config, **kw):
        captured["config"] = config
        return latent_image, latent_image

    mod = importlib.import_module("sampler_speed_sigma_harvest_node")
    original = mod.run_speed_pipeline
    mod.run_speed_pipeline = spy
    try:
        node = mod.MiniMaxH3SPEEDSigmaHarvest()
        node.sample(
            noise=EchoNoise(),
            guider=SpectralGuider(),
            sigmas=SIGMAS_10,
            latent_image=_latent(45, 80),
            **kwargs,
        )
    finally:
        mod.run_speed_pipeline = original

    harvest_cfg = captured["config"]
    for field in ("scales", "transition_steps", "transition_mode",
                  "noise_policy", "delta", "noise_amplitude",
                  "noise_decay_exponent", "transition_seed_offset",
                  "full_latent_h", "full_latent_w"):
        assert getattr(auto_cfg, field) == getattr(harvest_cfg, field), field
