"""I2V conditioning-latent resizing for multi-stage SPEED runs.

MiniMax-H3 stores keyframe and reference latents inside ``guider.original_conds``.
SPEED changes the video grid between stages, so keyframe latents must follow the
active spatial resolution. Reference latents intentionally remain untouched:
their packed-row layout is tied to the full-resolution model path.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch

log = logging.getLogger(__name__)


@dataclass
class _ConditionLatent:
    """One mutable keyframe holder plus its pristine full-resolution tensor."""

    holder: dict
    pristine: torch.Tensor

    @classmethod
    def from_holder(cls, holder: dict) -> "_ConditionLatent | None":
        latent = holder.get("latent") if isinstance(holder, dict) else None
        if latent is None or not hasattr(latent, "shape"):
            return None
        return cls(holder=holder, pristine=latent.clone())

    def resize(self, height: int, width: int) -> None:
        # MiniMax-H3 patchifies video on a 1x2x2 DiT grid. The main stage latent
        # is padded to that grid inside ComfyUI, but conditioning keyframes are
        # patchified directly without that padding. Round keyframe H/W up to the
        # same even grid here so their packed rows match the padded stage latent.
        height += height % 2
        width += width % 2
        current = self.holder.get("latent")
        if getattr(current, "shape", None) is not None and tuple(current.shape[-2:]) == (height, width):
            return
        self.holder["latent"] = _resize_condition(self.pristine, height, width)

    def restore(self) -> None:
        current = self.holder.get("latent")
        if getattr(current, "shape", None) != self.pristine.shape or not torch.equal(current, self.pristine):
            self.holder["latent"] = self.pristine.clone()


def _resize_condition(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize a 4D/5D conditioning latent spatially, frame by frame."""
    if latent.ndim == 4:
        if tuple(latent.shape[-2:]) == (height, width):
            return latent
        return torch.nn.functional.interpolate(
            latent,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=latent.dtype)

    if latent.ndim != 5:
        raise ValueError(f"unsupported cond latent ndim {latent.ndim} (want 4 or 5)")

    batch, channels, frames, source_h, source_w = latent.shape
    if (source_h, source_w) == (height, width):
        return latent
    flat = latent.transpose(1, 2).reshape(batch * frames, channels, source_h, source_w)
    resized = torch.nn.functional.interpolate(
        flat,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, channels, height, width).transpose(1, 2).to(dtype=latent.dtype)


class LatentWalker:
    """Own the keyframe snapshots for one generation.

    ``apply_stage()`` always resizes from pristine full-resolution tensors, so
    repeated stage changes never accumulate interpolation loss. ``apply_final()``
    restores the original keyframes and drops the snapshots. Reference latents
    are never wrapped or resized.
    """

    def __init__(self, guider):
        self._keyframes: dict[int, _ConditionLatent] = {}
        self._collect_keyframes(guider)

    def _collect_keyframes(self, guider) -> None:
        conds = getattr(guider, "original_conds", None)
        if not isinstance(conds, dict):
            return

        for cond_group in ("positive", "negative"):
            entries = conds.get(cond_group)
            if not isinstance(entries, list):
                continue
            for cond in entries:
                if not isinstance(cond, dict):
                    continue
                keyframes = cond.get("minimax_keyframes")
                if not isinstance(keyframes, list):
                    continue
                for holder in keyframes:
                    if not isinstance(holder, dict) or id(holder) in self._keyframes:
                        continue
                    wrapped = _ConditionLatent.from_holder(holder)
                    if wrapped is not None:
                        self._keyframes[id(holder)] = wrapped

    def apply_stage(self, height: int, width: int) -> None:
        """Resize every keyframe latent to the current coarse stage grid."""
        for wrapped in self._keyframes.values():
            before = getattr(wrapped.holder.get("latent"), "shape", None)
            wrapped.resize(height, width)
            after = getattr(wrapped.holder.get("latent"), "shape", None)
            if before != after:
                log.info(
                    "[LatentWalker] stage (%d,%d) — keyframe latent %s -> %s",
                    height,
                    width,
                    list(wrapped.pristine.shape),
                    list(after) if hasattr(after, "__iter__") else after,
                )

    def apply_final(self) -> None:
        """Restore pristine keyframes and release the generation snapshots."""
        for wrapped in self._keyframes.values():
            before = getattr(wrapped.holder.get("latent"), "shape", None)
            wrapped.restore()
            after = getattr(wrapped.holder.get("latent"), "shape", None)
            if before != after:
                log.info(
                    "[LatentWalker] final stage — restored keyframe latent %s",
                    list(after) if hasattr(after, "__iter__") else after,
                )
        self._keyframes.clear()


__all__ = ["LatentWalker"]
