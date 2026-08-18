"""Tests for debug/utility helper nodes.

Verifies:
- INPUT_TYPES is well-formed on each node
- FUNCTION runs on a mock nested latent without throwing
- Output shapes/types are reasonable
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

# --- Comfy stubs ---
_comfy = types.ModuleType("comfy")
for mod in ("samplers", "utils", "model_management", "nested_tensor"):
    sys.modules[f"comfy.{mod}"] = types.ModuleType(f"comfy.{mod}")
sys.modules["comfy"] = _comfy

# --- Mock H3 nested latent ---
class _MockNestedTensor:
    """Mock H3 nested latent with video + audio streams."""
    is_nested = True

    def __init__(self, video, audio):
        self._video = video
        self._audio = audio

    def unbind(self):
        return [self._video, self._audio]


def _make_mock_latent(H=32, W=32, T=8, C=4):
    """Create a synthetic nested latent (dict with 'samples' key)."""
    video = torch.randn(1, C, T, H, W)
    audio = torch.randn(1, C, 2, T)
    return {"samples": _MockNestedTensor(video, audio)}


# --- Helper to load a module by file path ---
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --- Load helper node modules directly ---
_NODE_DIR = Path(__file__).parent.parent.parent / "h3_speed_nodes" / "helper_nodes"

_mods = {}
for _name, _fn in [
    ("_node_inspect", "inspect.py"),
    ("_node_power_spectrum", "power_spectrum.py"),
    ("_node_dct_lowpass", "dct_lowpass.py"),
    ("_node_transition_math", "transition_math.py"),
    ("_node_spectral_expand", "spectral_expand.py"),
    ("_node_x0_fidelity_probe", "x0_fidelity_probe.py"),
    ("_node_av_reentry_oracle", "av_reentry_oracle.py"),
]:
    _mods[_name] = _load_module(_name, _NODE_DIR / _fn)

MiniMaxH3Inspect = _mods["_node_inspect"].MiniMaxH3Inspect
MiniMaxH3PowerSpectrum = _mods["_node_power_spectrum"].MiniMaxH3PowerSpectrum
MiniMaxH3DCTLowpass = _mods["_node_dct_lowpass"].MiniMaxH3DCTLowpass
MiniMaxH3TransitionMath = _mods["_node_transition_math"].MiniMaxH3TransitionMath
MiniMaxH3SpectralExpand = _mods["_node_spectral_expand"].MiniMaxH3SpectralExpand
MiniMaxH3XFidelityProbe = _mods["_node_x0_fidelity_probe"].MiniMaxH3XFidelityProbe
MiniMaxH3AVReentryOracle = _mods["_node_av_reentry_oracle"].MiniMaxH3AVReentryOracle


# --- INPUT_TYPES validation ---
@pytest.mark.parametrize(
    "node_cls",
    [
        MiniMaxH3Inspect,
        MiniMaxH3PowerSpectrum,
        MiniMaxH3DCTLowpass,
        MiniMaxH3TransitionMath,
        MiniMaxH3SpectralExpand,
        MiniMaxH3XFidelityProbe,
        MiniMaxH3AVReentryOracle,
    ],
)
def test_input_types_is_dict(node_cls):
    """INPUT_TYPES must be a dict with at least 'required' key."""
    types = node_cls.INPUT_TYPES()
    assert isinstance(types, dict)
    assert "required" in types
    assert isinstance(types["required"], dict)


# --- Function execution on mock nested latent ---
def test_inspect_node():
    """inspect node returns diagnostic string for nested latent."""
    node = MiniMaxH3Inspect()
    latent = _make_mock_latent(H=16, W=16, T=4)
    result = node.inspect(latent)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], str)
    assert "stream[0]" in result[0]
    assert "shape=(1, 4, 4, 16, 16)" in result[0]


def test_inspect_node_non_nested():
    """inspect node handles non-nested latent gracefully."""
    node = MiniMaxH3Inspect()
    latent = {"samples": torch.randn(1, 4, 4, 16, 16)}
    result = node.inspect(latent)
    assert isinstance(result, tuple)
    assert isinstance(result[0], str)


def test_power_spectrum_node():
    """power_spectrum node returns JSON string for nested latent."""
    node = MiniMaxH3PowerSpectrum()
    latent = _make_mock_latent(H=16, W=16, T=4)
    result = node.compute(latent)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], str)
    # Should contain JSON-like content
    assert "shape" in result[0]
    assert "freqs" in result[0]
    assert "power" in result[0]


def test_power_spectrum_node_empty():
    """power_spectrum node handles empty streams."""
    node = MiniMaxH3PowerSpectrum()

    class _EmptyNested:
        is_nested = True
        def unbind(self):
            return []

    latent = {"samples": _EmptyNested()}
    result = node.compute(latent)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert "No video stream" in result[0]


def test_dct_lowpass_node():
    """dct_lowpass node returns modified latent with low frequencies preserved."""
    node = MiniMaxH3DCTLowpass()
    latent = _make_mock_latent(H=16, W=16, T=4)
    result = node.apply(latent, cutoff_frequency=0.5)
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert "samples" in result[0]
    # Verify output is still nested
    samples = result[0]["samples"]
    assert hasattr(samples, "is_nested") and samples.is_nested
    streams = samples.unbind()
    assert len(streams) == 2  # video + audio
    # Video should be same shape
    assert streams[0].shape == (1, 4, 4, 16, 16)


def test_dct_lowpass_non_nested():
    """dct_lowpass node returns input unchanged for non-nested latent."""
    node = MiniMaxH3DCTLowpass()
    latent = {"samples": torch.randn(1, 4, 4, 16, 16)}
    result = node.apply(latent)
    assert isinstance(result, tuple)
    assert result[0] is latent  # same object back


def test_transition_math_node():
    """transition_math node returns (step, t_star, report)."""
    node = MiniMaxH3TransitionMath()
    result = node.compute(power_A=219.48, power_beta=2.42, delta=0.01, H=45, W=80, n_sigmas=20)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert isinstance(result[0], int)  # step
    assert isinstance(result[1], float)  # t_star
    assert isinstance(result[2], str)  # report
    assert "Transition step" in result[2]


def test_spectral_expand_node():
    """spectral_expand node returns expanded noise and report."""
    node = MiniMaxH3SpectralExpand()
    noise = _make_mock_latent(H=16, W=16, T=4)
    result = node.expand(noise, sigma=0.5, direction="up")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], dict)  # new noise
    assert isinstance(result[1], str)  # report
    assert "Expanded" in result[1]
    # Verify dimensions doubled
    expanded_video = result[0]["samples"].unbind()[0]
    assert expanded_video.shape[-2:] == (32, 32)  # 16x2 by 16x2


def test_spectral_expand_non_nested():
    """spectral_expand node handles non-nested noise."""
    node = MiniMaxH3SpectralExpand()
    noise = {"samples": torch.randn(1, 4, 4, 16, 16)}
    result = node.expand(noise, sigma=0.5, direction="up")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert "Not an H3 nested latent" in result[1]


def test_x0_fidelity_probe_node():
    """x0_fidelity_probe node returns fidelity score."""
    node = MiniMaxH3XFidelityProbe()
    x0 = _make_mock_latent(H=16, W=16, T=4)
    x_noisy = _make_mock_latent(H=16, W=16, T=4)
    result = node.probe(x0, x_noisy, sigma=0.5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)  # fidelity
    # Cosine similarity ranges from -1 to 1
    assert -1.0 <= result[0] <= 1.0
    assert isinstance(result[1], str)
    assert "X0 fidelity" in result[1]


def test_x0_fidelity_probe_non_nested():
    """x0_fidelity_probe node handles non-nested latent."""
    node = MiniMaxH3XFidelityProbe()
    x0 = {"samples": torch.randn(1, 4, 4, 16, 16)}
    x_noisy = {"samples": torch.randn(1, 4, 4, 16, 16)}
    result = node.probe(x0, x_noisy, sigma=0.5)
    assert isinstance(result, tuple)
    assert result[0] == 0.0
    assert "Not an H3 nested latent" in result[1]


def test_av_reentry_oracle_node():
    """av_reentry_oracle node finds reentry point."""
    node = MiniMaxH3AVReentryOracle()
    sigmas = torch.linspace(1.0, 0.01, 21)
    result = node.compute(sigmas, audio_shift=0.1)
    assert isinstance(result, tuple)
    assert len(result) == 2
    # result[0] is the original sigmas tensor (unchanged)
    assert hasattr(result[0], 'shape')  # it's a tensor
    assert isinstance(result[1], str)
    assert "re-enters at step" in result[1] or "reentry" in result[1].lower()


def test_av_reentry_oracle_no_reentry():
    """av_reentry_oracle node handles case where audio never re-enters."""
    node = MiniMaxH3AVReentryOracle()
    sigmas = torch.linspace(1.0, 0.5, 10)  # all above shift threshold
    result = node.compute(sigmas, audio_shift=0.1)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert "No reentry" in result[1]


# --- Duck-typed NestedTensor pass-through tests ---

class _DuckNestedTensor:
    """Simulates a real ComfyUI NestedTensor: accepts a list of streams."""

    is_nested = True

    def __init__(self, tensors):
        self._tensors = list(tensors)

    def unbind(self):
        return self._tensors


def _make_duck_latent(H=16, W=16, T=4, C=4):
    """Latent using a list-style NestedTensor (real ComfyUI convention)."""
    video = torch.randn(1, C, T, H, W)
    audio = torch.randn(1, C, 2, T)
    return {"samples": _DuckNestedTensor([video, audio])}


def test_dct_lowpass_preserves_nested_type():
    """DCT lowpass node returns the same NestedTensor subclass it receives."""
    latent = _make_duck_latent()
    original_cls = type(latent["samples"])
    node = _mods["_node_dct_lowpass"].MiniMaxH3DCTLowpass()
    result = node.apply(latent, cutoff_frequency=0.5)
    assert type(result[0]["samples"]) is original_cls


def test_spectral_expand_preserves_nested_type():
    """Spectral expand node returns the same NestedTensor subclass it receives."""
    latent = _make_duck_latent()
    original_cls = type(latent["samples"])
    node = _mods["_node_spectral_expand"].MiniMaxH3SpectralExpand()
    result = node.expand(latent, sigma=0.5, direction="up")
    assert type(result[0]["samples"]) is original_cls


def test_dct_lowpass_preserves_mock_fixture_type():
    """DCT lowpass also works with the (video, audio) fixture convention."""
    latent = _make_mock_latent()
    original_cls = type(latent["samples"])
    node = _mods["_node_dct_lowpass"].MiniMaxH3DCTLowpass()
    result = node.apply(latent, cutoff_frequency=0.5)
    assert type(result[0]["samples"]) is original_cls
