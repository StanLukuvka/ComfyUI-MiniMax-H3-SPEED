"""Latent with distinct stage boundaries — input → downsample → upsample → inject.

Each I2V keyframe latent is wrapped so its lifecycle is explicit and
cannot be accidentally re-used. The H3 runtime keeps one Latent per
keyframe id (and one RefLatent per ref2va block) in a tiny registry;
all resizes are driven from the *pristine* full-res clone.
"""

from __future__ import annotations

import enum
import torch


def _interp_cond(ic_z: torch.Tensor, ic_h: int, ic_w: int) -> torch.Tensor:
    """Per-frame bilinear spatial resize for cond latents [B,C,H,W] or [B,C,T,H,W].

    Duplicated here to keep latent self-contained and free of a circular
    import on h3_runtime. Behaviour is identical to h3_runtime._interp_cond.
    """
    if ic_z.ndim == 4:
        _, _, ic_hh, ic_ww = ic_z.shape
        if ic_hh == ic_h and ic_ww == ic_w:
            return ic_z
        return torch.nn.functional.interpolate(
            ic_z, size=(ic_h, ic_w), mode="bilinear", align_corners=False,
        ).to(dtype=ic_z.dtype)
    if ic_z.ndim != 5:
        raise ValueError(f"unsupported cond latent ndim {ic_z.ndim} (want 4 or 5)")
    ic_b, ic_c, ic_t, ic_hh, ic_ww = ic_z.shape
    if ic_hh == ic_h and ic_ww == ic_w:
        return ic_z
    ic_flat = ic_z.transpose(1, 2).reshape(ic_b * ic_t, ic_c, ic_hh, ic_ww)
    ic_resized = torch.nn.functional.interpolate(
        ic_flat, size=(ic_h, ic_w), mode="bilinear", align_corners=False,
    )
    return ic_resized.reshape(ic_b, ic_t, ic_c, ic_h, ic_w).transpose(1, 2).to(dtype=ic_z.dtype)


class LatentStage(enum.Enum):
    INPUT = "input"       # pristine captured, no stage yet
    STAGED = "staged"     # downscaled for a coarse stage
    INJECT = "inject"     # restored to full-res, ready for final stage
    CONSUMED = "consumed" # released, must not be reused


class Latent:
    """Owns one I2V keyframe latent lifecycle.

    Boundaries are intentionally narrow:
      input (pristine) -> scale_to(h,w) -> ... -> upscale_to_inject() -> release()

    Attributes:
      holder:      the keyframe dict this latent backs (so holder["latent"] stays in sync)
      pristine:    full-res clone captured at INPUT
      original_hw: (H, W) of pristine
      _current_hw: (H, W) of the live holder["latent"]
      stage:       LatentStage enum
    """

    def __init__(self, lh_holder: dict):
        if not isinstance(lh_holder, dict):
            raise TypeError("Latent holder must be a dict with a 'latent' tensor")
        lh_z = lh_holder.get("latent")
        if lh_z is None or not hasattr(lh_z, "shape"):
            raise TypeError("holder['latent'] must be a tensor with shape")
        # pristine is the ONLY source for every resize — never degrade
        self.holder = lh_holder
        self.pristine = lh_z.clone()
        self.original_hw: tuple[int, int] = (int(lh_z.shape[-2]), int(lh_z.shape[-1]))
        self._current_hw: tuple[int, int] = self.original_hw
        self.stage = LatentStage.INPUT
        self._injected = False

    @property
    def current_hw(self) -> tuple[int, int]:
        return self._current_hw

    @property
    def is_consumed(self) -> bool:
        return self.stage == LatentStage.CONSUMED

    def scale_to(self, st_h: int, st_w: int) -> torch.Tensor:
        """Downsample boundary — even-round, same-size skip, INPUT/STAGED only."""
        if self.stage == LatentStage.CONSUMED:
            raise RuntimeError("Latent already consumed — cannot scale")
        if self._injected:
            raise RuntimeError("Latent already injected — cannot scale")
        # DiT 2x2 patch grid: odd dims would crash patchify_video, so round UP to even.
        st_th = st_h + (st_h % 2)
        st_tw = st_w + (st_w % 2)
        if self._current_hw == (st_th, st_tw):
            self.stage = LatentStage.STAGED
            return self.holder["latent"]
        # Always resize from pristine, never from the live (degraded) tensor.
        self.holder["latent"] = _interp_cond(self.pristine, st_th, st_tw)
        self._current_hw = (st_th, st_tw)
        self.stage = LatentStage.STAGED
        return self.holder["latent"]

    def restore(self) -> torch.Tensor:
        """Upsample/inject boundary — restore pristine full-res if needed.

        Called at the final full-res stage. If the live tensor is not already
        full-res, it is replaced with a clone of pristine. Either way the
        stage becomes INJECT.
        """
        if self.stage == LatentStage.CONSUMED:
            return self.holder.get("latent", self.pristine)
        r_z = self.holder.get("latent")
        if getattr(r_z, "shape", None) != getattr(self.pristine, "shape", None):
            self.holder["latent"] = self.pristine.clone()
            self._current_hw = self.original_hw
        self.stage = LatentStage.INJECT
        self._injected = True
        return self.holder["latent"]

    # upscale_to_inject is the richer name from the design doc — alias to restore
    def upscale_to_inject(self) -> torch.Tensor:  # type: ignore[no-redef]
        return self.restore()

    def release(self) -> None:
        """CONSUMED boundary — drop refs so the full-res clone dies with generation."""
        self.stage = LatentStage.CONSUMED
        # holder is kept but stage guards prevent reuse; caller pops registry entry.
        # We do not clear holder["latent"] — the live latent is still needed by model.

    def inject_payload(self) -> torch.Tensor:
        """Compatibility alias — same as restore()."""
        return self.restore()


class RefLatent(Latent):
    """Variant for ref2va blocks — never scales.

    ref2va rows are allocated from stored latent_h/latent_w metadata which
    SPEED never changes; tensor and allocation stay consistent at every stage.
    We still track lifecycle so the pristine clone has a bounded lifetime,
    but scale_to is a no-op.
    """

    def scale_to(self, h: int, w: int) -> torch.Tensor:  # type: ignore[override]
        if self.stage == LatentStage.CONSUMED:
            raise RuntimeError("RefLatent already consumed")
        if self._injected:
            raise RuntimeError("RefLatent already injected")
        # Intentionally no resize — keep full-res at every coarse stage.
        self.stage = LatentStage.STAGED
        return self.holder["latent"]


__all__ = ["Latent", "RefLatent", "LatentStage", "_interp_cond"]
