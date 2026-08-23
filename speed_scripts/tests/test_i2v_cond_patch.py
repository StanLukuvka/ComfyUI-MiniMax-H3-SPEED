"""Regression tests for the per-stage I2V keyframe rescale (h3_runtime).

Covers the two failure modes the 5-attempt saga ended with:
- the multi-keyframe restore bug (final stage must restore EVERY keyframe,
  not just the first), and
- the pristine-store leak (entries must be popped at final-stage restore).
"""
import pytest
import torch

from speed_scripts.h3_runtime import (
    _PRISTINE_STORE,
    _rescale_cond_latents,
)


def _kf(h, w):
    return {"resolved_frame_index": 0, "latent": torch.rand(1, 1, 2, h, w)}


@pytest.fixture(autouse=True)
def _clear_pristine_store():
    """Each test gets a clean pristine store (it is keyed by id(), so stale
    entries from other tests would alias or pollute length assertions)."""
    _PRISTINE_STORE.clear()
    yield
    _PRISTINE_STORE.clear()


def test_coarse_stage_resizes_from_pristine_and_never_degrades():
    cond = {"minimax_keyframes": [_kf(8, 16)]}
    # Stage 0: downscale twice with different targets — the second resize must
    # come from the pristine snapshot, not the first resize (no degradation).
    _rescale_cond_latents(cond, 3, 5, is_final_stage=False)
    first = cond["minimax_keyframes"][0]["latent"].clone()
    _rescale_cond_latents(cond, 5, 9, is_final_stage=False)
    second = cond["minimax_keyframes"][0]["latent"]
    # targets rounded up to even: (4, 6) and (6, 10)
    assert tuple(first.shape[-2:]) == (4, 6)
    assert tuple(second.shape[-2:]) == (6, 10)
    pristine = _PRISTINE_STORE.get(id(cond["minimax_keyframes"][0]))
    assert pristine is not None
    assert tuple(pristine.shape[-2:]) == (8, 16), "pristine must stay full-res"


def test_final_stage_restores_all_keyframes_and_pops_store():
    kf1, kf2 = _kf(8, 16), _kf(8, 16)
    cond = {"minimax_keyframes": [kf1, kf2]}
    # Coarse pass resizes both.
    _rescale_cond_latents(cond, 5, 9, is_final_stage=False)
    assert len(_PRISTINE_STORE) == 2
    assert tuple(kf1["latent"].shape[-2:]) == (6, 10)

    # Final stage must restore BOTH keyframes (regression: the old code
    # returned after the first keyframe) and drop the snapshots.
    _rescale_cond_latents(cond, 8, 16, is_final_stage=True)
    assert tuple(kf1["latent"].shape[-2:]) == (8, 16)
    assert tuple(kf2["latent"].shape[-2:]) == (8, 16)
    assert not _PRISTINE_STORE, "pristine snapshots must be released at final stage"


def test_same_size_stage_keeps_tensor_and_final_stage_releases_it():
    kf = _kf(8, 16)
    cond = {"minimax_keyframes": [kf]}
    before = kf["latent"]
    _rescale_cond_latents(cond, 8, 16, is_final_stage=False)
    assert kf["latent"] is before
    # First touch always snapshots pristine (so a later coarse stage has a
    # source), even if this stage needed no resize.
    assert _PRISTINE_STORE.get(id(kf)) is not None

    # Final stage: tensor must be the SAME object (already full-res) and the
    # snapshot must be released.
    _rescale_cond_latents(cond, 8, 16, is_final_stage=True)
    assert kf["latent"] is before
    assert not _PRISTINE_STORE


def test_negative_conds_are_walked_too():
    # parity with _patch_guider_conditioning_for_stage: negative and positive
    # conds are both walked.
    from speed_scripts.h3_runtime import _patch_guider_conditioning_for_stage

    class FakeGuider:
        original_conds = {
            "positive": [{"minimax_keyframes": [_kf(8, 16)]}],
            "negative": [{"minimax_keyframes": [_kf(8, 16)]}],
        }

    g = FakeGuider()
    _patch_guider_conditioning_for_stage(g, 5, 9, is_final_stage=False)
    for conds in g.original_conds.values():
        assert tuple(conds[0]["minimax_keyframes"][0]["latent"].shape[-2:]) == (6, 10)
