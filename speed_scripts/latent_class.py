"""LatentClass + LatentWalker — the I2V condition-latent lifecycle.

ComfyUI's H3 I2V path stashes per-keyframe and per-ref latents on the
conditioning dict (under `minimax_keyframes` and `minimax_refs`). SPEED
denoises at multiple spatial resolutions in sequence, so the *stored*
sources must be re-resized at every stage boundary — full-res tensor +
coarse grid is the 520-vs-130 row-mismatch crash.

Per-holder lifecycle (LatentClass):
  input (pristine) -> downscale(h, w) -> ... -> upscale_to_inject() -> release()

Per-generation orchestrator (LatentWalker):
  Construct once with a guider, call apply_stage() at every stage boundary,
  discard the walker when the generation is done. All wrapped latents die
  with it. Two walkers in flight never see each other's state because the
  dict is per-instance, not class-level.
"""

from __future__ import annotations

import enum
import logging

import torch

log = logging.getLogger(__name__)


class LatentStage(enum.Enum):
    INPUT = "input"
    STAGED = "staged"
    INJECT = "inject"
    CONSUMED = "consumed"


class LatentClass:
    """One I2V keyframe or ref latent lifecycle.

    Mutates `holder["latent"]` in place as stages run, so the model sees
    a latent that matches the current stage grid. The pristine full-res
    snapshot is captured at INPUT and never leaves this instance — it dies
    when the LatentWalker that owns this wrapper is garbage-collected.

    Refs (ref2va blocks) opt out of downscaling: their row allocation is
    locked to full res by the model and resizing them breaks PackedLayout.
    """

    def __init__(self, holder: dict, *, is_ref: bool = False):
        if not isinstance(holder, dict):
            raise TypeError("LatentClass holder must be a dict with a 'latent' tensor")
        lc_z = holder.get("latent")
        if lc_z is None or not hasattr(lc_z, "shape"):
            raise TypeError("holder['latent'] must be a tensor with shape")
        self.holder = holder
        self.is_ref = is_ref
        # pristine is the ONLY source for every resize — never degrade.
        self.pristine = lc_z.clone()
        self.original_hw: tuple[int, int] = (int(lc_z.shape[-2]), int(lc_z.shape[-1]))
        self.current_hw: tuple[int, int] = self.original_hw
        self.stage = LatentStage.INPUT
        self._injected = False

    @property
    def is_consumed(self) -> bool:
        return self.stage == LatentStage.CONSUMED

    def downscale(self, d_h: int, d_w: int) -> torch.Tensor:
        """Downsample boundary — even-round, same-size skip, INPUT/STAGED only."""
        if self.stage == LatentStage.CONSUMED:
            raise RuntimeError("latent already consumed — cannot downscale")
        if self._injected:
            raise RuntimeError("latent already injected — cannot downscale")
        if self.is_ref:
            self.stage = LatentStage.STAGED
            return self.holder["latent"]
        # DiT 2x2 patch grid: odd dims would crash patchify_video, so round UP to even.
        d_th = d_h + (d_h % 2)
        d_tw = d_w + (d_w % 2)
        if self.current_hw == (d_th, d_tw):
            self.stage = LatentStage.STAGED
            return self.holder["latent"]
        # Always resize from pristine, never from the live (degraded) tensor.
        self.holder["latent"] = self._resize_cond(self.pristine, d_th, d_tw)
        self.current_hw = (d_th, d_tw)
        self.stage = LatentStage.STAGED
        return self.holder["latent"]

    def upscale_to_inject(self) -> torch.Tensor:
        """Upsample/inject boundary — restore pristine full-res if needed."""
        if self.stage == LatentStage.CONSUMED:
            return self.holder.get("latent", self.pristine)
        if not self.is_ref:
            uti_z = self.holder.get("latent")
            if getattr(uti_z, "shape", None) != getattr(self.pristine, "shape", None):
                self.holder["latent"] = self.pristine.clone()
                self.current_hw = self.original_hw
        self.stage = LatentStage.INJECT
        self._injected = True
        return self.holder["latent"]

    def release(self) -> None:
        """CONSUMED boundary — drop refs so the full-res clone dies with generation."""
        self.stage = LatentStage.CONSUMED

    @staticmethod
    def _resize_cond(rc_z: torch.Tensor, rc_h: int, rc_w: int) -> torch.Tensor:
        """Per-frame bilinear spatial resize for cond latents.

        torch.nn.functional.interpolate is 4D-native (size=(h,w) on a 5D tensor
        raises), so fold the temporal axis into the batch for the resize and
        fold it back out after — frames are interpolated independently.

        Lives on LatentClass for namespacing: nothing outside this class
        resizes cond latents.
        """
        if rc_z.ndim == 4:
            rc_hh, rc_ww = rc_z.shape[-2], rc_z.shape[-1]
            if rc_hh == rc_h and rc_ww == rc_w:
                return rc_z
            rc_resized = torch.nn.functional.interpolate(
                rc_z, size=(rc_h, rc_w), mode="bilinear", align_corners=False,
            )
            return rc_resized.to(dtype=rc_z.dtype)
        if rc_z.ndim != 5:
            raise ValueError(f"unsupported cond latent ndim {rc_z.ndim} (want 4 or 5)")
        rc_b, rc_c, rc_t, rc_hh, rc_ww = rc_z.shape
        if rc_hh == rc_h and rc_ww == rc_w:
            return rc_z
        rc_flat = rc_z.transpose(1, 2).reshape(rc_b * rc_t, rc_c, rc_hh, rc_ww)
        rc_resized = torch.nn.functional.interpolate(
            rc_flat, size=(rc_h, rc_w), mode="bilinear", align_corners=False,
        )
        return rc_resized.reshape(rc_b, rc_t, rc_c, rc_h, rc_w).transpose(1, 2).to(dtype=rc_z.dtype)


class LatentWalker:
    """Per-generation orchestrator for I2V condition-latent resizing.

    Construct once with a guider. The constructor snapshots pristine for
    every keyframe + ref on the guider. Call `apply_stage(h, w)` at every
    coarse stage boundary, then `apply_final()` at the full-res stage. The
    walker owns all wrapped LatentClass instances — they die when it does.

    Two walkers in flight never see each other's state because the wrapper
    dict is per-instance, not class-level.
    """

    def __init__(self, w_guider):
        self._wrappers: dict[int, LatentClass] = {}
        self._prime(w_guider)

    def _get(self, gw_holder, *, is_ref: bool) -> LatentClass | None:
        """Get-or-create the wrapper for one holder dict. None if holder has no latent tensor."""
        gw_existing = self._wrappers.get(id(gw_holder))
        if gw_existing is not None:
            return gw_existing
        gw_z = gw_holder.get("latent")
        if gw_z is None or not hasattr(gw_z, "shape"):
            return None
        gw_wrapped = LatentClass(gw_holder, is_ref=is_ref)
        self._wrappers[id(gw_holder)] = gw_wrapped
        return gw_wrapped

    def _release(self, rh_holder) -> None:
        """Release + pop the wrapper for one holder. Safe if no wrapper exists."""
        rh_wrapped = self._wrappers.pop(id(rh_holder), None)
        if rh_wrapped is not None:
            rh_wrapped.release()

    def _prime(self, p_guider) -> None:
        """Snapshot pristine for every keyframe/ref on a guider. No resizing.

        Call once from __init__ so the wrapper dict is populated and a
        downscale at the first stage boundary can resize from pristine.
        """
        p_conds_box = getattr(p_guider, "original_conds", None)
        if not isinstance(p_conds_box, dict):
            return
        for p_conds_key in ("positive", "negative"):
            p_conds_list = p_conds_box.get(p_conds_key)
            if not isinstance(p_conds_list, list):
                continue
            for p_cond in p_conds_list:
                if not isinstance(p_cond, dict):
                    continue
                p_kfs = p_cond.get("minimax_keyframes")
                if isinstance(p_kfs, list):
                    for p_kf in p_kfs:
                        if isinstance(p_kf, dict):
                            self._get(p_kf, is_ref=False)
                p_refs = p_cond.get("minimax_refs")
                if isinstance(p_refs, list):
                    for p_ref in p_refs:
                        if isinstance(p_ref, dict):
                            self._get(p_ref, is_ref=True)

    def apply_stage(self, as_h: int, as_w: int) -> None:
        """Resize every wrapped keyframe latent to (as_h, as_w) for one coarse stage.

        Refs are intentionally not rescaled: their row allocation is locked
        to full res by the model and resizing breaks PackedLayout.
        """
        for as_wrapped in self._wrappers.values():
            if as_wrapped.is_ref:
                continue
            as_before = getattr(as_wrapped.holder.get("latent"), "shape", None)
            try:
                as_wrapped.downscale(as_h, as_w)
            except RuntimeError:
                continue
            as_after = getattr(as_wrapped.holder.get("latent"), "shape", None)
            if as_before != as_after:
                log.info(
                    "[LatentWalker] stage (%d,%d) — keyframe latent %s -> %s",
                    as_h, as_w,
                    list(as_wrapped.pristine.shape),
                    list(as_after) if hasattr(as_after, "__iter__") else as_after,
                )

    def apply_final(self) -> None:
        """Restore every wrapped latent to full-res and release the wrapper.

        Call once at the final full-res stage. After this, the walker is
        inert — every wrapper has been released.
        """
        for af_wrapped in self._wrappers.values():
            af_before = getattr(af_wrapped.holder.get("latent"), "shape", None)
            try:
                af_wrapped.upscale_to_inject()
            except RuntimeError:
                continue
            af_after = getattr(af_wrapped.holder.get("latent"), "shape", None)
            if not af_wrapped.is_ref and af_before != af_after:
                log.info(
                    "[LatentWalker] final stage — restored keyframe latent %s",
                    list(af_after) if hasattr(af_after, "__iter__") else af_after,
                )
        # Drop every wrapper — full-res clones die with this walker.
        for af_holder_id in list(self._wrappers):
            af_wrapped = self._wrappers.pop(af_holder_id, None)
            if af_wrapped is not None:
                af_wrapped.release()


__all__ = [
    "LatentClass",
    "LatentWalker",
    "LatentStage",
]
