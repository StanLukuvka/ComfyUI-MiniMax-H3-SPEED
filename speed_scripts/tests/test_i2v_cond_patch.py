"""Regression tests for the per-stage I2V keyframe rescale (h3_runtime).

Covers the two failure modes the 5-attempt saga ended with:
- the multi-keyframe restore bug (final stage must restore EVERY keyframe,
  not just the first), and
- the pristine-store leak (entries must be popped at final-stage restore).
"""
import pytest
import torch

from speed_scripts.latent_class import LatentWalker


def _kf(h, w):
    return {"resolved_frame_index": 0, "latent": torch.rand(1, 1, 2, h, w)}


def _walker(cond):
    """Build a one-cond walker for tests, no real guider needed."""
    class _G:
        original_conds = {"positive": [cond], "negative": []}
    return LatentWalker(_G())


def test_coarse_stage_resizes_from_pristine_and_never_degrades():
    cond = {"minimax_keyframes": [_kf(8, 16)]}
    w = _walker(cond)
    # Stage 0: downscale twice with different targets — the second resize must
    # come from the pristine snapshot, not the first resize (no degradation).
    w.apply_stage(3, 5)
    first = cond["minimax_keyframes"][0]["latent"].clone()
    w.apply_stage(5, 9)
    second = cond["minimax_keyframes"][0]["latent"]
    # targets rounded up to even: (4, 6) and (6, 10)
    assert tuple(first.shape[-2:]) == (4, 6)
    assert tuple(second.shape[-2:]) == (6, 10)
    pristine = w._wrappers[id(cond["minimax_keyframes"][0])].pristine
    assert pristine is not None
    assert tuple(pristine.shape[-2:]) == (8, 16), "pristine must stay full-res"


def test_final_stage_restores_all_keyframes_and_drops_wrappers():
    kf1, kf2 = _kf(8, 16), _kf(8, 16)
    cond = {"minimax_keyframes": [kf1, kf2]}
    w = _walker(cond)
    # Coarse pass resizes both.
    w.apply_stage(5, 9)
    assert len(w._wrappers) == 2
    assert tuple(kf1["latent"].shape[-2:]) == (6, 10)

    # Final stage must restore BOTH keyframes (regression: the old code
    # returned after the first keyframe) and drop the snapshots.
    w.apply_final()
    assert tuple(kf1["latent"].shape[-2:]) == (8, 16)
    assert tuple(kf2["latent"].shape[-2:]) == (8, 16)
    assert not w._wrappers, "wrappers must be released at final stage"


def test_same_size_stage_keeps_tensor_and_final_stage_releases_it():
    kf = _kf(8, 16)
    cond = {"minimax_keyframes": [kf]}
    before = kf["latent"]
    w = _walker(cond)
    w.apply_stage(8, 16)
    assert kf["latent"] is before
    # First touch always snapshots pristine (so a later coarse stage has a
    # source), even if this stage needed no resize.
    assert w._wrappers[id(kf)].pristine is not None

    # Final stage: tensor must be the SAME object (already full-res) and the
    # wrapper must be released.
    w.apply_final()
    assert kf["latent"] is before
    assert not w._wrappers


def test_negative_conds_are_walked_too():
    # Parity with the old per-stage patch: negative and positive conds are
    # both walked by LatentWalker.
    pos = {"minimax_keyframes": [_kf(8, 16)]}
    neg = {"minimax_keyframes": [_kf(8, 16)]}

    class _G:
        original_conds = {"positive": [pos], "negative": [neg]}

    w = LatentWalker(_G())
    w.apply_stage(5, 9)
    for conds in (pos, neg):
        assert tuple(conds["minimax_keyframes"][0]["latent"].shape[-2:]) == (6, 10)
