"""LatentClass — owns the I2V condition-latent lifecycle.

ComfyUI's H3 I2V path stashes per-keyframe and per-ref latents on the
conditioning dict (under `minimax_keyframes` and `minimax_refs`). SPEED
denoises at multiple spatial resolutions in sequence, so the *stored*
sources must be re-resized at every stage boundary — full-res tensor +
coarse grid is the 520-vs-130 row-mismatch crash.

Boundaries:
  input (pristine) -> downscale(h, w) -> ... -> upscale_to_inject() -> release()

The class also owns the id-keyed registry of every wrapped holder, the
cond-walk (mix), the guider-walk (walk_guider), and the one-shot pristine
priming (prime).
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


def _resize_cond(rc_z: torch.Tensor, rc_h: int, rc_w: int) -> torch.Tensor:
    """Per-frame bilinear spatial resize for cond latents.

    torch.nn.functional.interpolate is 4D-native (size=(h,w) on a 5D tensor
    raises), so fold the temporal axis into the batch for the resize and fold
    it back out after — frames are interpolated independently.
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


class LatentClass:
    """One I2V keyframe or ref latent lifecycle + the shared registry.

    Constructed from a `holder` dict that has a `"latent"` tensor. The wrapper
    mutates `holder["latent"]` in place as stages run, so the model sees a
    latent that matches the current stage grid.

    Refs (ref2va blocks) opt out of downscaling by passing `is_ref=True`;
    their row allocation is locked to full res by the model and resizing them
    breaks PackedLayout.

    The class also owns the id-keyed registry of every wrapped holder. Use
    the classmethods `for_holder`, `release_holder`, `mix`, `walk_guider`,
    `prime`, `clear` to operate across holders.
    """

    # Plain keyframe/ref dicts refuse setattr, so the id()-keyed class dict
    # is the only store. Entries are popped when release() fires so the
    # full-res clones die with each generation.
    _registry: dict[int, "LatentClass"] = {}

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

    # -- introspection ------------------------------------------------------
    @property
    def is_consumed(self) -> bool:
        return self.stage == LatentStage.CONSUMED

    # -- per-instance boundaries ---------------------------------------------
    def downscale(self, d_h: int, d_w: int) -> torch.Tensor:
        """Downsample boundary — even-round, same-size skip, INPUT/STAGED only.

        Refs are intentionally not rescaled: their rows are allocated from
        stored latent_h/latent_w metadata, which SPEED never changes.
        """
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
        self.holder["latent"] = _resize_cond(self.pristine, d_th, d_tw)
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

    # -- class-level registry ops -------------------------------------------
    @classmethod
    def for_holder(cls, fh_holder, *, is_ref: bool) -> "LatentClass | None":
        """Get or create the wrapper for one holder dict. None if holder has no latent tensor."""
        fh_existing = cls._registry.get(id(fh_holder))
        if fh_existing is not None:
            return fh_existing
        fh_z = fh_holder.get("latent")
        if fh_z is None or not hasattr(fh_z, "shape"):
            return None
        fh_wrapped = cls(fh_holder, is_ref=is_ref)
        cls._registry[id(fh_holder)] = fh_wrapped
        return fh_wrapped

    @classmethod
    def release_holder(cls, rh_holder) -> None:
        """Release + pop the wrapper for one holder. Safe if no wrapper exists."""
        rh_wrapped = cls._registry.pop(id(rh_holder), None)
        if rh_wrapped is not None:
            rh_wrapped.release()

    @classmethod
    def clear(cls) -> None:
        """Drop every wrapped latent. Call between generations or in test teardown."""
        cls._registry.clear()

    @classmethod
    def prime(cls, p_guider) -> None:
        """Snapshot pristine for every keyframe/ref on a guider. No resizing.

        Call once before the first stage so the registry is populated and a
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
                for p_kf in p_cond.get("minimax_keyframes", []) or []:
                    if isinstance(p_kf, dict):
                        cls.for_holder(p_kf, is_ref=False)
                for p_ref in p_cond.get("minimax_refs", []) or []:
                    if isinstance(p_ref, dict):
                        cls.for_holder(p_ref, is_ref=True)

    @classmethod
    def mix(cls, m_cond: dict, m_h: int, m_w: int, *, is_final_stage: bool = False) -> None:
        """Walk one cond and rescale every fl2va keyframe + ref2va ref.

        Refs are intentionally not rescaled. At is_final_stage, restore
        pristine on every source and release the wrapper.
        """
        if not isinstance(m_cond, dict):
            return
        # --- keyframes (fl2va) ---
        for m_kf in m_cond.get("minimax_keyframes", []) or []:
            m_wrapped = cls.for_holder(m_kf, is_ref=False)
            if m_wrapped is None:
                continue
            if is_final_stage:
                m_before = getattr(m_kf.get("latent"), "shape", None)
                m_wrapped.upscale_to_inject()
                m_after = getattr(m_kf.get("latent"), "shape", None)
                if m_before != m_after:
                    log.info(
                        "[LatentClass] final stage — restored keyframe latent %s",
                        list(m_after) if hasattr(m_after, "__iter__") else m_after,
                    )
                cls.release_holder(m_kf)
                continue
            m_before = getattr(m_kf.get("latent"), "shape", None)
            try:
                m_wrapped.downscale(m_h, m_w)
            except RuntimeError:
                continue
            m_after = getattr(m_kf.get("latent"), "shape", None)
            if m_before != m_after:
                log.info(
                    "[LatentClass] stage (%d,%d) — keyframe latent %s -> %s",
                    m_h, m_w,
                    list(m_wrapped.pristine.shape),
                    list(m_after) if hasattr(m_after, "__iter__") else m_after,
                )
        # --- refs (ref2va) ---
        for m_ref in m_cond.get("minimax_refs", []) or []:
            m_wrapped = cls.for_holder(m_ref, is_ref=True)
            if m_wrapped is None:
                continue
            if is_final_stage:
                try:
                    m_wrapped.upscale_to_inject()
                except RuntimeError:
                    pass
                cls.release_holder(m_ref)
                continue
            try:
                m_wrapped.downscale(m_h, m_w)
            except RuntimeError:
                continue

    @classmethod
    def walk_guider(cls, wg_guider, wg_h: int, wg_w: int, *, is_final_stage: bool = False) -> None:
        """Apply mix to every conditioning in wg_guider.original_conds (positive + negative)."""
        wg_conds_box = getattr(wg_guider, "original_conds", None)
        if not isinstance(wg_conds_box, dict):
            return
        for wg_conds_key in ("positive", "negative"):
            wg_conds_list = wg_conds_box.get(wg_conds_key)
            if not isinstance(wg_conds_list, list):
                continue
            for wg_cond in wg_conds_list:
                if not isinstance(wg_cond, dict):
                    continue
                cls.mix(wg_cond, wg_h, wg_w, is_final_stage=is_final_stage)


__all__ = [
    "LatentClass",
    "LatentStage",
]
