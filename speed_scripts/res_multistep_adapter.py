"""Stateful RES Multistep adapter for the SPEED pipeline.

Deterministic, non-ancestral RES Multistep (``eta=0``): no noise injection,
no SDE, no CFG++ path. Original implementation written from the published
RES second-order multistep formulation (exponential integrators in
``t = -ln(sigma)`` space). ComfyUI is used only as the behavioral reference
for the host sampler-function contract — its solver source is never copied.

Why this exists: the native ``res_multistep`` sampler keeps its step history
in function locals, so every SPEED stage call would restart with empty
history. Worse, the native code derives the previous input sigma from
``sigmas[i - 1]``, which is the wrong element on the first step of a
stage-local schedule. This adapter owns that history explicitly in a
``ResMultistepState`` that the SPEED runtime can carry across stages
according to the selected boundary policy, and reads ``t_prev`` from the
state instead of the schedule.

``reset`` is the default boundary policy. ``projected`` is
the historical project-and-rebase comparison path. Experimental V3.0
``hybrid`` applies one spatial-only VIDEO DCT candidate mix and uses the
first-order AUDIO candidate. Temporal transitions clear history instead of
using that operator. Flat host tensors fail closed when recorded stream shape
metadata is absent or inconsistent. These semantics are covered by automated
tests only; this module does not claim GPU or native ComfyUI validation.
"""

# FLOW-PRODUCED: RES candidate-helper refactor.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .flow import aligned_sigma
from .spectral import dct2, dct_temporal, idct2, idct_temporal, spectral_expand_clean_3d


def _host_ksampler_class():
    """The host's real ``KSAMPLER`` class, resolved lazily.

    Importing it at module load would break ComfyUI-free environments (the
    surrounding package and its tests must import without a full ComfyUI
    install, which only provides ``comfy.samplers`` at runtime). ``None``
    means no real host is installed; then only the sampler-function contract
    is available and the object must not pretend otherwise.
    """
    try:
        from comfy.samplers import KSAMPLER
    except Exception:
        return None
    return KSAMPLER


@dataclass
class ResMultistepState:
    """History maintained by one RES sampler run; the run-scoped handle owns
    the policy for what survives a SPEED boundary.

    ``old_denoised`` is the previous model denoised estimate, ``old_sigma_down``
    the previous interval's destination sigma, and ``prev_sigma_in`` the
    previous interval's input sigma. All three are written after every
    completed interval; a single-sigma stage executes zero intervals and
    leaves them untouched.
    """

    old_denoised: object | None = None
    old_sigma_down: float | None = None
    prev_sigma_in: float | None = None
    hybrid_pending: bool = False
    hybrid_second_order_thw: tuple[int, int, int] | None = None
    hybrid_target_stream_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    def clear(self) -> None:
        """Release every previous-step history reference."""
        self.old_denoised = None
        self.old_sigma_down = None
        self.prev_sigma_in = None
        self.hybrid_pending = False
        self.hybrid_second_order_thw = None
        self.hybrid_target_stream_shapes = None


def project_clean_history(history, target_thw: tuple[int, int, int], source_stream_shapes=None):
    """DCT-project stored denoised history to the target video geometry.

    ``history`` is the previous denoised estimate stored in
    ``ResMultistepState.old_denoised``. Only the video geometry is projected,
    with :func:`spectral_expand_clean_3d`: the stored estimate receives no
    fresh high-frequency noise and is not scaled by sigma. The audio slice is
    copied unchanged by this historical projected-mode operation.

    This helper defines the mechanics of the historical experimental
    ``projected`` boundary mode. It does not establish equivalence to an
    uninterrupted target-resolution RES trajectory. Native RES has no
    geometry-changing boundary to provide such an oracle.

    Two shapes arrive here, depending on where the history was captured:

    * the nested H3 video/audio pair (video ``[B,C,T,H,W]`` + audio
      ``[B,C,2,T_audio]``) a guider hands back when it executes the sampler
      directly; or
    * the flat tensor the real host produces at the sampler boundary. The
      host ``CFGGuider.sample`` packs nested video+audio with
      ``comfy.utils.pack_latents`` (each stream reshaped ``[B, 1, -1]``, then
      concatenated on the last axis) before any sampler code runs, so the
      model's denoised output — and therefore this history — is a plain
      ``[B, 1, N]`` tensor. ``source_stream_shapes`` carries the per-stream
      shapes from that pack; the video slice is the first
      ``prod(video_shape[1:])`` elements and the audio slice is the rest.
      Missing or inconsistent metadata raises ``ValueError``. The helper
      never infers a stream split from the flat element count.
    """
    if getattr(history, "is_nested", False):
        streams = list(history.unbind())
        if len(streams) != 2:
            raise ValueError("RES clean history must contain exactly video and audio streams")
        video, audio = streams
    elif torch.is_tensor(history) and history.ndim == 3:
        if not source_stream_shapes or len(source_stream_shapes) != 2:
            raise ValueError(
                "flat RES clean history needs the host pack's per-stream shapes"
            )
        shapes = [tuple(int(d) for d in shape) for shape in source_stream_shapes]
        if sum(math.prod(s[1:]) for s in shapes) != history.shape[-1]:
            raise ValueError(
                "flat RES clean history does not match the recorded stream shapes"
            )
        cut = math.prod(shapes[0][1:])
        video = history[:, :, :cut].reshape(shapes[0])
        audio = history[:, :, cut:].reshape(shapes[1])
    else:
        raise ValueError(
            "RES clean history must be a nested H3 video/audio pair or the "
            "host's flat packed tensor"
        )
    projected_video = spectral_expand_clean_3d(video, target_thw)
    if getattr(history, "is_nested", False):
        return type(history)([projected_video, audio])
    # Re-pack in the host pack_latents layout: [B, 1, N] slices concatenated
    # on the last axis, matching the tensor shape the next host stage consumes.
    batch = projected_video.shape[0]
    return torch.cat(
        (
            projected_video.reshape(batch, 1, -1),
            audio.reshape(batch, 1, -1),
        ),
        dim=-1,
    )


def rebase_res_history_sigmas(
    state: ResMultistepState,
    new_sigma: float,
    ratio: float,
) -> None:
    """Apply the historical projected-mode sigma-coordinate transform.

    ``projected`` mode sets the previous interval destination to the aligned
    next-stage boundary and maps ``prev_sigma_in`` through the same
    ``aligned_sigma`` transform used by SPEED's boundary. This function
    defines that experimental boundary behavior; native RES itself does not
    define how previous-step sigma metadata should be transformed when latent
    geometry changes. Empty fields stay empty. Mutates ``state`` in place and
    never touches the scheduler or working sigma schedule.
    """
    if state.old_sigma_down is not None:
        state.old_sigma_down = float(new_sigma)
    if state.prev_sigma_in is not None:
        prev = float(state.prev_sigma_in)
        if prev == 1.0:
            # aligned_sigma rejects q >= 1, while an input sigma of exactly
            # 1.0 is legal previous-step metadata. Applying the transform's
            # formula at q=1 gives kappa=1, so the historical projected-mode
            # mapping leaves that coordinate unchanged.
            state.prev_sigma_in = prev
        else:
            _, state.prev_sigma_in = aligned_sigma(prev, ratio)


def _res_first_order_update(x, denoised, sigma, sigma_down):
    """Return the existing first-order RES candidate for one interval."""
    d = (x - denoised) / sigma
    return x + d * (sigma_down - sigma)


def _res_second_order_update(
    x,
    denoised,
    old_denoised,
    sigma_f,
    sigma_down_f,
    old_sigma_down,
    prev_sigma_in,
):
    """Return the existing second-order RES candidate for one interval."""
    t_old = -math.log(old_sigma_down)
    t_next = -math.log(sigma_down_f)
    t_prev = -math.log(prev_sigma_in)
    h = t_next + math.log(sigma_f)
    c2 = (t_prev - t_old) / h
    phi1 = math.expm1(-h) / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    b1 = 0.0 if math.isnan(phi1 - phi2 / c2) else phi1 - phi2 / c2
    b2 = 0.0 if math.isnan(phi2 / c2) else phi2 / c2
    return math.exp(-h) * x + h * (b1 * denoised + b2 * old_denoised)


def _hybridize_video(first_video, second_video, source_thw):
    """Replace the source geometry coefficient block in a VIDEO candidate."""
    source_t, source_h, source_w = source_thw
    target_t, target_h, target_w = first_video.shape[-3:]
    if any(source > target for source, target in zip(
        source_thw, (target_t, target_h, target_w),
    )):
        raise ValueError("RES hybrid source geometry cannot exceed target VIDEO geometry")
    mixed_coeff = dct2(dct_temporal(first_video))
    mixed_coeff[..., :source_t, :source_h, :source_w] = dct2(
        dct_temporal(second_video)
    )[..., :source_t, :source_h, :source_w]
    return idct_temporal(idct2(mixed_coeff)).to(dtype=first_video.dtype)


def hybridize_res_candidates(
    first_order,
    second_order,
    second_order_thw: tuple[int, int, int],
    target_stream_shapes=None,
):
    """Blend VIDEO candidates in DCT space and retain first-order AUDIO.

    The accepted inputs are the nested H3 video/audio pair or the flat
    ``[B, 1, N]`` pack produced by the real host. Flat packs require the
    host-recorded stream shapes because their boundary does not describe the
    split itself. This is the experimental V3.0 one-interval operator for
    spatial-only transitions. Missing or inconsistent metadata fails closed.
    """
    source_t, source_h, source_w = (int(v) for v in second_order_thw)
    if min(source_t, source_h, source_w) < 1:
        raise ValueError("RES hybrid source geometry must be positive")

    nested = getattr(first_order, "is_nested", False)
    if nested != getattr(second_order, "is_nested", False):
        raise ValueError("RES hybrid candidates must use the same representation")
    if nested:
        first_streams = list(first_order.unbind())
        second_streams = list(second_order.unbind())
        if len(first_streams) != 2 or len(second_streams) != 2:
            raise ValueError("RES hybrid candidates must contain exactly video and audio")
        first_video, first_audio = first_streams
        second_video, second_audio = second_streams
        if first_video.shape != second_video.shape or first_audio.shape != second_audio.shape:
            raise ValueError("RES hybrid candidate stream shapes must agree")
        if first_video.ndim != 5 or first_audio.ndim != 4:
            raise ValueError("RES hybrid nested candidates must be H3 video and audio")
        mixed_video = _hybridize_video(
            first_video, second_video, (source_t, source_h, source_w),
        )
        return type(first_order)([mixed_video, first_audio])

    if not (torch.is_tensor(first_order) and torch.is_tensor(second_order)):
        raise ValueError("RES hybrid candidates must be nested pairs or flat tensors")
    if first_order.ndim != 3 or first_order.shape[1] != 1:
        raise ValueError("flat RES hybrid candidates must have shape [B, 1, N]")
    if second_order.shape != first_order.shape:
        raise ValueError("RES hybrid candidate pack shapes must agree")
    if not target_stream_shapes or len(target_stream_shapes) != 2:
        raise ValueError("flat RES hybrid candidates need target stream shapes")
    shapes = [tuple(int(d) for d in shape) for shape in target_stream_shapes]
    if len(shapes[0]) != 5 or len(shapes[1]) != 4:
        raise ValueError("target stream shapes must describe H3 video and audio")
    if any(shape[0] != first_order.shape[0] for shape in shapes):
        raise ValueError("target stream shapes must match the packed batch")
    video_count = math.prod(shapes[0][1:])
    audio_count = math.prod(shapes[1][1:])
    if video_count + audio_count != first_order.shape[-1]:
        raise ValueError("target stream shapes do not match the packed candidates")
    first_video = first_order[..., :video_count].reshape(shapes[0])
    second_video = second_order[..., :video_count].reshape(shapes[0])
    first_audio = first_order[..., video_count:].reshape(shapes[1])
    mixed_video = _hybridize_video(
        first_video, second_video, (source_t, source_h, source_w),
    )
    return torch.cat(
        (mixed_video.reshape(first_order.shape[0], 1, -1), first_audio.reshape(first_order.shape[0], 1, -1)),
        dim=-1,
    )


def res_multistep_sampler(
    model,
    noise,
    sigmas,
    state: ResMultistepState,
    extra_args=None,
    callback=None,
    disable=None,
):
    """Run deterministic RES Multistep over ``sigmas``, carrying ``state``.

    Matches the host ``KSAMPLER`` sampler-FUNCTION contract: called as
    ``fn(model, noise, sigmas, extra_args=..., callback=..., disable=...)``
    where ``model(x, sigma, **extra_args)`` returns the denoised estimate
    and ``callback`` receives the native per-step dict. ``disable`` is
    accepted for contract compatibility; this implementation has no progress
    bar of its own. (The sampler-OBJECT contract is ``.sample`` on
    :class:`ResMultistepSampler`; the two are not interchangeable.)
    """
    extra_args = {} if extra_args is None else extra_args
    x = noise
    s_in = x.new_ones([x.shape[0]]) if torch.is_tensor(x) else 1.0

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        denoised = model(x, sigma * s_in, **extra_args)
        # Deterministic RES (eta=0): the destination sigma is the next
        # schedule entry and no noise is ever added.
        sigma_down = sigmas[i + 1]
        if callback is not None:
            callback({
                "x": x, "i": i, "sigma": sigma, "sigma_hat": sigma,
                "denoised": denoised,
            })

        sigma_f = float(sigma)
        sigma_down_f = float(sigma_down)
        old_denoised = state.old_denoised
        old_sigma_down = state.old_sigma_down
        prev_sigma_in = state.prev_sigma_in
        # First order when the destination sigma is zero, when the carried
        # history is missing, or when the interval is degenerate (zero width
        # or a zero previous-step gap, which would divide by zero below).
        # For every well-formed strictly decreasing schedule this reduces to
        # the plan rule: first order iff destination sigma is zero or
        # old_denoised is missing.
        use_second_order = (
            0.0 < sigma_down_f < sigma_f
            and old_denoised is not None
            and old_sigma_down is not None
            and prev_sigma_in is not None
            and old_sigma_down != prev_sigma_in
        )
        if use_second_order:
            # Second order RES multistep: exponential integrators in
            # t = -ln(sigma) space. t_prev comes from the carried state,
            # never from sigmas[i - 1]: on the first step of a stage-local
            # schedule that index is the wrong element.
            x_first = _res_first_order_update(x, denoised, sigma, sigma_down)
            x_second = _res_second_order_update(
                x, denoised, old_denoised,
                sigma_f, sigma_down_f, old_sigma_down, prev_sigma_in,
            )
            if state.hybrid_pending and state.hybrid_second_order_thw is not None:
                x = hybridize_res_candidates(
                    x_first,
                    x_second,
                    state.hybrid_second_order_thw,
                    state.hybrid_target_stream_shapes,
                )
            else:
                x = x_second
        else:
            # First order (Euler).
            x = _res_first_order_update(x, denoised, sigma, sigma_down)

        state.old_denoised = denoised
        state.old_sigma_down = sigma_down_f
        state.prev_sigma_in = sigma_f
        if state.hybrid_pending:
            state.hybrid_pending = False
            state.hybrid_second_order_thw = None
            state.hybrid_target_stream_shapes = None
    return x


class ResMultistepSampler:
    """Host sampler object wrapping the stateful RES function.

    The host guider invokes a sampler *object* as
    ``sampler.sample(guider, sigmas, extra_args, callback, noise,
    latent_image, denoise_mask, disable_pbar)`` (``CFGGuider.inner_sample``
    wraps ``sampler.sample`` in its wrapper executor), which is the contract
    native ``KSAMPLER`` objects implement. This class therefore wraps the
    host's own ``KSAMPLER`` — constructed with :func:`res_multistep_sampler`
    as its sampler function — and delegates ``.sample`` to it, so
    ``noise_scaling``, the inpaint model wrapper, the per-step callback
    adaptation, and ``inverse_noise_scaling`` all stay host-native (plan §13:
    do not bypass ``guider.sample()``).

    ``__call__`` is the separate sampler-*function* contract: it runs
    :func:`res_multistep_sampler` directly on already-host-processed inputs,
    as a plain ``KSAMPLER`` sampler function would receive them. The two
    contracts are not interchangeable; only ``.sample`` satisfies the host
    sampler-object seam.
    """

    def __init__(self, state: ResMultistepState | None = None):
        self.state = state if state is not None else ResMultistepState()
        ksampler_cls = _host_ksampler_class()
        if ksampler_cls is None:
            raise RuntimeError(
                "ResMultistepSampler requires the host comfy.samplers.KSAMPLER; "
                "no real ComfyUI host is available in this environment"
            )
        self._ksampler = ksampler_cls(self._run)

    def _run(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        return res_multistep_sampler(
            model, noise, sigmas, self.state,
            extra_args=extra_args, callback=callback, disable=disable,
        )

    def sample(self, model_wrap, sigmas, extra_args, callback, noise,
               latent_image=None, denoise_mask=None, disable_pbar=False):
        return self._ksampler.sample(
            model_wrap, sigmas, extra_args, callback, noise,
            latent_image=latent_image, denoise_mask=denoise_mask,
            disable_pbar=disable_pbar,
        )

    def __call__(self, model, noise, sigmas, extra_args=None, callback=None, disable=None):
        return self._run(
            model, noise, sigmas,
            extra_args=extra_args, callback=callback, disable=disable,
        )
