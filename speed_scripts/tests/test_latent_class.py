"""LatentClass tests — the new abstraction that owns the cond-latent lifecycle.

The old `Latent` / `RefLatent` data classes still exist in
`speed_scripts.latent` for backwards compat; the new `LatentClass` is the
canonical owner of pristine snapshots, the registry, and the cond walk.
"""

import torch

from speed_scripts.latent_class import (
    LatentClass,
    LatentStage,
)


def _kf(h, w):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def _ref(h, w):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def setup_function(_):
    LatentClass.clear()


def teardown_function(_):
    LatentClass.clear()


def test_latent_class_input_to_staged_rounds_odd_dims_up_to_even():
    kf = _kf(8, 16)
    lc = LatentClass(kf)
    assert lc.stage == LatentStage.INPUT
    assert lc.current_hw == (8, 16)

    # odd dims -> rounded UP to even (4, 6)
    lc.downscale(3, 5)
    assert tuple(kf["latent"].shape[-2:]) == (4, 6)
    assert lc.stage == LatentStage.STAGED


def test_latent_class_downscale_uses_pristine_not_live():
    """Two consecutive downscales must both come from the pristine snapshot."""
    kf = _kf(8, 16)
    lc = LatentClass(kf)
    lc.downscale(3, 5)  # (4, 6)
    lc.downscale(5, 9)  # (6, 10) — must not be a resize of (4, 6)
    assert tuple(kf["latent"].shape[-2:]) == (6, 10)
    # pristine is unchanged
    assert tuple(lc.pristine.shape[-2:]) == (8, 16)


def test_latent_class_upscale_to_inject_restores_pristine():
    kf = _kf(8, 16)
    lc = LatentClass(kf)
    lc.downscale(3, 5)  # (4, 6)
    lc.upscale_to_inject()
    assert tuple(kf["latent"].shape[-2:]) == (8, 16)
    assert lc.stage == LatentStage.INJECT


def test_latent_class_release_blocks_further_changes():
    kf = _kf(8, 16)
    lc = LatentClass(kf)
    lc.release()
    assert lc.is_consumed
    # release() returns the holder's current latent (or pristine fallback),
    # but does not re-shape it.
    out = lc.upscale_to_inject()
    assert tuple(out.shape[-2:]) == (8, 16)


def test_ref_variant_never_rescales():
    """RefLatent (via is_ref=True) tracks lifecycle but does not resize."""
    ref = _ref(8, 16)
    lc = LatentClass(ref, is_ref=True)
    lc.downscale(2, 2)
    # Ref stays at full res.
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)
    assert lc.stage == LatentStage.STAGED


def test_mix_walks_positive_and_negative_conds():
    """LatentClass.mix() handles positive + negative, keyframes + refs."""
    kf_p = _kf(8, 16)
    kf_n = _kf(8, 16)
    ref_p = _ref(8, 16)
    cond_pos = {"minimax_keyframes": [kf_p], "minimax_refs": [ref_p]}
    cond_neg = {"minimax_keyframes": [kf_n], "minimax_refs": []}
    # No `original_conds` wrapping — feed `mix` directly.
    LatentClass.mix(cond_pos, 3, 5, is_final_stage=False)
    LatentClass.mix(cond_neg, 3, 5, is_final_stage=False)
    assert tuple(kf_p["latent"].shape[-2:]) == (4, 6)
    assert tuple(kf_n["latent"].shape[-2:]) == (4, 6)
    # Refs stay full res.
    assert tuple(ref_p["latent"].shape[-2:]) == (8, 16)


def test_mix_final_stage_restores_and_releases():
    kf = _kf(8, 16)
    cond = {"minimax_keyframes": [kf], "minimax_refs": []}
    LatentClass.mix(cond, 3, 5, is_final_stage=False)
    assert tuple(kf["latent"].shape[-2:]) == (4, 6)
    assert len(LatentClass._registry) == 1
    LatentClass.mix(cond, 8, 16, is_final_stage=True)
    assert tuple(kf["latent"].shape[-2:]) == (8, 16)
    # Registry entry popped after release.
    assert len(LatentClass._registry) == 0


def test_prime_populates_registry_without_resizing():
    """prime() snapshots pristine but does not resize any holder."""
    kf = _kf(8, 16)
    ref = _ref(8, 16)
    cond = {"minimax_keyframes": [kf], "minimax_refs": [ref]}

    class _FakeGuider:
        original_conds = {"positive": [cond], "negative": []}

    g = _FakeGuider()
    LatentClass.prime(g)
    assert len(LatentClass._registry) == 2
    # Latents unchanged.
    assert tuple(kf["latent"].shape[-2:]) == (8, 16)
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)


def test_walk_guider_applies_mix_to_positive_and_negative():
    kf_p = _kf(8, 16)
    kf_n = _kf(8, 16)
    cond_p = {"minimax_keyframes": [kf_p], "minimax_refs": []}
    cond_n = {"minimax_keyframes": [kf_n], "minimax_refs": []}

    class _FakeGuider:
        original_conds = {"positive": [cond_p], "negative": [cond_n]}

    g = _FakeGuider()
    LatentClass.walk_guider(g, 3, 5, is_final_stage=False)
    assert tuple(kf_p["latent"].shape[-2:]) == (4, 6)
    assert tuple(kf_n["latent"].shape[-2:]) == (4, 6)


def test_clear_drops_all_wrapped_latents():
    kf = _kf(8, 16)
    cond = {"minimax_keyframes": [kf], "minimax_refs": []}
    LatentClass.mix(cond, 3, 5, is_final_stage=False)
    assert len(LatentClass._registry) == 1
    LatentClass.clear()
    assert len(LatentClass._registry) == 0
