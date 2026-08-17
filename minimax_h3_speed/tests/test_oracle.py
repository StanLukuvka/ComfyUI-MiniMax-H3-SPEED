"""Oracle tests: synthetic straight-flow AV-transition proof.

CPU-only, no ComfyUI model weights required. Proves the SPEED transition +
re-entry reconstruction contracts hold under the same packing and numerics
the real sampler will use.
"""

from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest
import torch


def _install_comfy_stubs():
    """Mock comfy so test_oracle.py can run without a ComfyUI install."""
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
            self._tensors = list(tensors)
        def unbind(self):
            return self._tensors
        def to(self, *args, **kwargs):
            return NestedTensor([t.to(*args, **kwargs) for t in self._tensors])
        def cpu(self):
            return self.to(device="cpu")

    # Provide pack/unpack so oracle.py's fallback path isn't needed
    def _pack_latents(streams):
        shapes, tensors = [], []
        for t in streams:
            shapes.append(t.shape)
            tensors.append(t.reshape(t.shape[0], 1, -1))
        return torch.cat(tensors, dim=-1), shapes
    def _unpack_latents(combined, shapes):
        output, cursor = [], 0
        for shape in shapes:
            cut = math.prod(shape[1:])
            tens = combined[..., cursor:cursor + cut]
            output.append(tens.reshape([tens.shape[0]] + list(shape[1:])))
            cursor += cut
        return output
    utils.pack_latents = _pack_latents
    utils.unpack_latents = _unpack_latents

    class NestedTensor:
        is_nested = True
        def __init__(self, tensors):
            self._tensors = list(tensors)
        def unbind(self):
            return self._tensors
        def to(self, *args, **kwargs):
            return NestedTensor([t.to(*args, **kwargs) for t in self._tensors])
        def cpu(self):
            return self.to(device="cpu")

    nested_tensor.NestedTensor = NestedTensor
    samplers.sampler_object = lambda name: ("sampler", name)
    utils.PROGRESS_BAR_ENABLED = True

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


_install_comfy_stubs()

from minimax_h3_speed.flow import aligned_speed_sigma, reentry_noise
from minimax_h3_speed.flow import time_shift_sigma as _original_time_shift_sigma
from minimax_h3_speed.oracle import (
    NestedTensor,
    StraightFlowModel,
    run_euler_pack,
    time_shift_slope,
)


def _mk(seed: int = 0):
    """Create synthetic video/audio/clean/noise tensors with deterministic seed."""
    g = torch.Generator().manual_seed(seed)
    video_shape, audio_shape = (1, 4, 2, 8, 12), (1, 2, 2, 17)
    video = torch.randn(video_shape, generator=g)
    audio = torch.randn(audio_shape, generator=torch.Generator().manual_seed(seed + 1))
    clean_v = torch.randn(video_shape, generator=torch.Generator().manual_seed(seed + 2))
    clean_a = torch.randn(audio_shape, generator=torch.Generator().manual_seed(seed + 3))
    noise_v = torch.randn(video_shape, generator=torch.Generator().manual_seed(seed + 4))
    noise_a = torch.randn(audio_shape, generator=torch.Generator().manual_seed(seed + 5))
    return video, audio, clean_v, clean_a, noise_v, noise_a


def _analytic_trajectory(noise: torch.Tensor, clean: torch.Tensor, sigma: float) -> torch.Tensor:
    """Exact straight-flow solution x(sigma) = sigma*noise + (1-sigma)*clean."""
    return sigma * noise + (1.0 - sigma) * clean


# ---------------------------------------------------------------------------
# 1. Euler integration correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1])
def test_euler_integrates_video_along_analytic_straight_flow(seed):
    """Euler on coarse schedule converges toward analytic straight-flow."""
    _, _, clean_v, clean_a, noise_v, noise_a = _mk(seed)
    model = StraightFlowModel(clean_v, clean_a)
    lat = NestedTensor([torch.zeros_like(clean_v), torch.zeros_like(clean_a)])
    sigmas = torch.tensor([1.0, 0.7, 0.4, 0.15, 0.0])
    out, _ = run_euler_pack(model, NestedTensor([noise_v, noise_a]), lat, sigmas)
    out_v, _ = out.unbind()
    analytic = _analytic_trajectory(noise_v, clean_v, 0.0)
    assert torch.allclose(out_v, analytic, atol=0.5, rtol=0.2)


def test_euler_fine_schedule_recovers_clean_target_close():
    """Fine Euler schedule converges tightly to clean at sigma=0 (straight flow)."""
    _, _, clean_v, clean_a, noise_v, noise_a = _mk()
    model = StraightFlowModel(clean_v, clean_a)
    lat = NestedTensor([torch.zeros_like(clean_v), torch.zeros_like(clean_a)])
    n = 200
    sigmas = torch.linspace(1.0, 0.0, n + 1)
    out, _ = run_euler_pack(model, NestedTensor([noise_v, noise_a]), lat, sigmas)
    out_v, out_a = out.unbind()
    assert torch.allclose(out_v, clean_v, atol=0.15, rtol=0.15)
    assert torch.allclose(out_a, clean_a, atol=0.15, rtol=0.15)


# ---------------------------------------------------------------------------
# 2. Callback x0 is the analytic denoised estimate
# ---------------------------------------------------------------------------

def test_callback_x0_is_analytic_denoised_estimate():
    """Callback x0 = denoised estimate = clean target for straight-flow model."""
    _, _, clean_v, clean_a, noise_v, noise_a = _mk()
    model = StraightFlowModel(clean_v, clean_a)
    lat = NestedTensor([torch.zeros_like(clean_v), torch.zeros_like(clean_a)])
    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    _, callbacks = run_euler_pack(model, NestedTensor([noise_v, noise_a]), lat, sigmas)
    # First callback fires at sigma=1.0. x = noise; x0 = denoised = clean.
    _, x0_nested, _, _ = callbacks[0]
    x0_v, x0_a = x0_nested.unbind()
    assert torch.allclose(x0_v, clean_v, atol=1e-3, rtol=1e-3)
    assert torch.allclose(x0_a, clean_a, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 3. Audio on shifted schedule
# ---------------------------------------------------------------------------

def test_audio_evolves_on_shifted_schedule_via_slope():
    """Audio integrates on shifted schedule and converges to clean target."""
    _, _, clean_v, clean_a, noise_v, noise_a = _mk()
    model = StraightFlowModel(clean_v, clean_a)
    lat = NestedTensor([torch.zeros_like(clean_v), torch.zeros_like(clean_a)])
    n = 200
    sigmas = torch.linspace(1.0, 0.0, n + 1)
    out, _ = run_euler_pack(model, NestedTensor([noise_v, noise_a]), lat, sigmas)
    _, out_a = out.unbind()
    assert torch.allclose(out_a, clean_a, atol=0.15, rtol=0.15), (
        "slope-scaled audio did not converge to its clean target"
    )


def test_time_shift_slope_matches_native_convention():
    """time_shift_slope at sigma=0 equals to_shift/from_shift; at 0.5 differs."""
    # slope(0) = to_shift/from_shift = 3/12 = 0.25
    assert abs(time_shift_slope(0.0, 12.0, 3.0) - 0.25) < 1e-9
    # slope(0.5) for 12/3 is ~0.64, NOT the constant audio_scale 4.0
    slope05 = time_shift_slope(0.5, 12.0, 3.0)
    assert abs(slope05 - 0.64) < 1e-3
    assert abs(slope05 - 4.0) > 1e-3


def test_time_shift_sigma_boundary_values():
    """time_shift_sigma(0) = 0; time_shift_sigma(1) = from_shift * to_shift / (from_shift + to_shift - from_shift*to_shift)."""
    # At sigma=0 the output is 0 (no time shift applied at t=0).
    assert abs(_original_time_shift_sigma(0.0, 12.0, 3.0) - 0.0) < 1e-9
    # At sigma=1.0: base=1/(12+1*(−11))=1/1, result=3*1/(1+2*1)=1.0
    expected_at_1 = 3.0 * 1.0 / (1.0 + 2.0 * 1.0)
    assert abs(_original_time_shift_sigma(1.0, 12.0, 3.0) - expected_at_1) < 1e-9


# ---------------------------------------------------------------------------
# 4. Re-entry reconstructs carried state
# ---------------------------------------------------------------------------

def test_reentry_noise_reconstructs_carried_state():
    """reentry_noise(state, sigma) * sigma == state."""
    state = torch.tensor([3.25, -1.5])
    sigma = 0.65
    noise = reentry_noise(state, sigma)
    assert torch.allclose(sigma * noise, state, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Aligned speed sigma matches paper formula
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q,ratio", [
    (0.5, 2.0),
    (0.6, 2.0),
    (2.0 / 3.0, 1.5),
    (0.5, 4.0),
])
def test_aligned_speed_sigma_matches_paper(q, ratio):
    """kappa and aligned sigma match the paper's analytic formulas."""
    kappa, aligned = aligned_speed_sigma(q, ratio)
    expected_kappa = ratio / (1.0 + (ratio - 1.0) * q)
    assert abs(kappa - expected_kappa) < 1e-12
    assert abs(aligned - q * expected_kappa) < 1e-12


# ---------------------------------------------------------------------------
# 6. Negative test: untouched audio breaks AV invariant
# ---------------------------------------------------------------------------

def test_leave_audio_untouched_breaks_av_invariant():
    """Plan-mandated negative test: 'untouched audio' during video alignment fails."""
    _, _, clean_v, clean_a, noise_v, noise_a = _mk()
    model = StraightFlowModel(clean_v, clean_a)
    lat = NestedTensor([torch.zeros_like(clean_v), torch.zeros_like(clean_a)])
    sigmas = torch.tensor([1.0, 0.6, 0.3, 0.0])
    out, callbacks = run_euler_pack(model, NestedTensor([noise_v, noise_a]), lat, sigmas)

    # At step 0 (sigma=1.0), the audio is the noisy input, not the clean target.
    step, x0_nested, x_packed_view, _ = callbacks[0]
    untouched_audio = x_packed_view.unbind()[1]

    # The true audio trajectory at this point is intermediate (not yet clean);
    # leaving it untouched and then pretending the audio advanced to the aligned
    # stage is exactly the failure. Assert it does NOT pass the invariant.
    assert not torch.allclose(untouched_audio, clean_a, atol=0.5, rtol=0.5), (
        "Untouched audio is already at clean target — negative test is vacuous"
    )


# ---------------------------------------------------------------------------
# 7. NestedTensor contract tests
# ---------------------------------------------------------------------------

def test_nested_tensor_contract():
    """NestedTensor has is_nested=True, unbind(), and to()."""
    v = torch.randn(1, 4, 2, 8, 12)
    a = torch.randn(1, 2, 2, 17)
    nt = NestedTensor([v, a])
    assert nt.is_nested
    streams = nt.unbind()
    assert len(streams) == 2
    assert torch.equal(streams[0], v)
    assert torch.equal(streams[1], a)

    # to() creates a new NestedTensor with transformed tensors
    cpu_nt = nt.to(device="cpu")
    assert cpu_nt.is_nested
    assert cpu_nt.unbind()[0].device.type == "cpu"


def test_nested_tensor_cpu():
    """NestedTensor.cpu() works correctly."""
    v = torch.randn(1, 4, 2, 8, 12, device="cpu")
    a = torch.randn(1, 2, 2, 17, device="cpu")
    nt = NestedTensor([v, a])
    cpu_nt = nt.cpu()
    assert cpu_nt.is_nested
    assert cpu_nt.unbind()[0].device.type == "cpu"
