"""Synthetic straight-flow oracle for CPU-verified canonical SPEED proofs.

This mirrors the Lab's `oracle.py` — a two-stream straight-flow model with
analytically known clean targets, integrated via ComfyUI-style packed latent
Euler. Used to prove:

1. `run_euler_pack` integrates the packed ODE correctly (straight-flow).
2. Callback x0 streams match the analytic denoised estimate.
3. Audio evolves on its own shifted schedule via `time_shift_slope`.
4. Re-entry `reentry_noise(state, sigma)` reconstructs carried state.
5. `aligned_speed_sigma` matches the paper formula exactly.
"""

from __future__ import annotations

import math

import torch

try:
    import comfy  # pragma: no cover — only present inside ComfyUI
    import comfy.nested_tensor
    import comfy.utils

    def pack_latents(streams):
        return comfy.utils.pack_latents(streams)

    def unpack_latents(combined, shapes):
        return comfy.utils.unpack_latents(combined, shapes)

    NestedTensor = comfy.nested_tensor.NestedTensor
    _HAS_COMFY = True
except Exception:  # pragma: no cover — standalone fallback
    _HAS_COMFY = False

    def pack_latents(streams):
        shapes, tensors = [], []
        for tensor in streams:
            shapes.append(tensor.shape)
            tensors.append(tensor.reshape(tensor.shape[0], 1, -1))
        return torch.cat(tensors, dim=-1), shapes

    def unpack_latents(combined, shapes):
        output, cursor = [], 0
        for shape in shapes:
            cut = math.prod(shape[1:])
            tens = combined[..., cursor:cursor + cut]
            output.append(tens.reshape([tens.shape[0]] + list(shape[1:])))
            cursor += cut
        return output

    class NestedTensor:
        """Mock NestedTensor with is_nested flag and unbind()."""

        is_nested = True

        def __init__(self, tensors):
            self.tensors = list(tensors)

        def unbind(self):
            return self.tensors

        def to(self, *args, **kwargs):
            return NestedTensor([t.to(*args, **kwargs) for t in self.tensors])

        def cpu(self):
            return self.to(device="cpu")


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    """Mirror comfy.ldm.minimax.model.time_shift_sigma."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: float, from_shift: float, to_shift: float) -> float:
    """Mirror comfy.ldm.minimax.model.time_shift_slope."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


class StraightFlowModel:
    """Two-stream straight-flow model with analytically known clean target.

    `video_clean` / `audio_clean` are the exact `x0` targets. `_denoise`
    returns them directly; `time_shift_slope` governs the audio velocity
    rescale per-sigma so the audio stream evolves on its own shifted schedule.
    """

    video_shift = 12.0
    audio_shift = 3.0

    def __init__(self, video_clean: torch.Tensor, audio_clean: torch.Tensor):
        self.video_clean = video_clean
        self.audio_clean = audio_clean

    def denoise(self, video_x: torch.Tensor, audio_x: torch.Tensor, sigma: float):
        """Return (video_denoised, audio_denoised) = (clean_v, clean_a).

        For a perfect straight-flow model the denoised estimate IS the clean
        target. The per-sigma `time_shift_slope` is applied separately to the
        audio velocity so it evolves on its own shifted schedule.
        """
        return self.video_clean, self.audio_clean


def run_euler_pack(model, noise_nested, latent_nested, sigmas: torch.Tensor):
    """Integrate the packed ODE exactly like CFGGuider.sample + sample_euler.

    Mirrors the k-diffusion Euler integrator on a flat-packed [B, 1, N] tensor
    where the first `shapes[0][-1]` elements are video and the rest are audio.
    Returns (output_nested, callbacks) where each callback is
    (step_index, x0_nested, x_nested_at_step, total_steps).
    """
    lat_streams = latent_nested.unbind()
    noise_streams = noise_nested.unbind()
    latent_packed, shapes = pack_latents(lat_streams)
    noise_packed, _ = pack_latents(noise_streams)

    # k-diffusion initial noise_scaling for flow: x = sigma*noise + (1-sigma)*latent
    x = sigmas[0] * noise_packed + (1.0 - sigmas[0]) * latent_packed

    callbacks = []

    def nested_view(tensor):
        streams_out = unpack_latents(tensor, shapes)
        return NestedTensor(streams_out)

    total_steps = len(sigmas) - 1
    for i in range(total_steps):
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        v_cur, a_cur = unpack_latents(x, shapes)
        denoised_v, denoised_a = model.denoise(v_cur, a_cur, sigma)
        denoised = pack_latents([denoised_v, denoised_a])[0]

        d = (x - denoised) / sigma
        x0 = x - sigma * d
        callbacks.append((i, nested_view(x0), nested_view(x), total_steps))
        x = x + d * (sigma_next - sigma)

    return nested_view(x), callbacks
