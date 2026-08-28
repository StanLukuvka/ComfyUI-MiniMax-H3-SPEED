"""LatentClass + LatentWalker tests.

LatentClass is the per-holder lifecycle (one instance per keyframe/ref).
LatentWalker is the per-generation orchestrator — construct it once with a
guider, call apply_stage() at every coarse boundary, apply_final() at the
end. Each instance owns its own wrapper dict, so two walkers never collide.
"""

import torch

from speed_scripts.latent_class import (
    LatentClass,
    LatentStage,
    LatentWalker,
)


def _kf(h, w):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def _ref(h, w):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def _guider(*conds):
    """Build a fake guider whose original_conds has every cond as positive."""
    class _G:
        original_conds = {"positive": list(conds), "negative": []}
    return _G()


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
    """is_ref=True tracks lifecycle but does not resize."""
    ref = _ref(8, 16)
    lc = LatentClass(ref, is_ref=True)
    lc.downscale(2, 2)
    # Ref stays at full res.
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)
    assert lc.stage == LatentStage.STAGED


def test_walker_resizes_keyframes_in_every_cond():
    """LatentWalker applies the resize to every positive+negative cond."""
    kf_p = _kf(8, 16)
    kf_n = _kf(8, 16)
    ref_p = _ref(8, 16)
    cond_pos = {"minimax_keyframes": [kf_p], "minimax_refs": [ref_p]}
    cond_neg = {"minimax_keyframes": [kf_n], "minimax_refs": []}
    g = _guider(cond_pos)
    # Inject the negative cond by editing the same fake object.
    g.original_conds["negative"] = [cond_neg]
    w = LatentWalker(g)
    w.apply_stage(3, 5)
    assert tuple(kf_p["latent"].shape[-2:]) == (4, 6)
    assert tuple(kf_n["latent"].shape[-2:]) == (4, 6)
    # Refs stay full res.
    assert tuple(ref_p["latent"].shape[-2:]) == (8, 16)


def test_walker_final_stage_restores_and_releases():
    kf = _kf(8, 16)
    cond = {"minimax_keyframes": [kf], "minimax_refs": []}
    w = LatentWalker(_guider(cond))
    w.apply_stage(3, 5)
    assert tuple(kf["latent"].shape[-2:]) == (4, 6)
    assert len(w._wrappers) == 1
    w.apply_final()
    assert tuple(kf["latent"].shape[-2:]) == (8, 16)
    # Every wrapper popped on final.
    assert len(w._wrappers) == 0


def test_walker_prime_populates_wrappers_without_resizing():
    """Constructor snapshots pristine but does not resize any holder."""
    kf = _kf(8, 16)
    ref = _ref(8, 16)
    cond = {"minimax_keyframes": [kf], "minimax_refs": [ref]}
    w = LatentWalker(_guider(cond))
    assert len(w._wrappers) == 2
    # Latents unchanged.
    assert tuple(kf["latent"].shape[-2:]) == (8, 16)
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)


def test_two_walkers_dont_collide():
    """Per-instance wrapper dicts isolate independent generations."""
    kf_a = _kf(8, 16)
    kf_b = _kf(8, 16)
    cond_a = {"minimax_keyframes": [kf_a], "minimax_refs": []}
    cond_b = {"minimax_keyframes": [kf_b], "minimax_refs": []}
    walker_a = LatentWalker(_guider(cond_a))
    walker_b = LatentWalker(_guider(cond_b))
    walker_a.apply_stage(3, 5)
    # walker_a resized, walker_b untouched.
    assert tuple(kf_a["latent"].shape[-2:]) == (4, 6)
    assert tuple(kf_b["latent"].shape[-2:]) == (8, 16)
    # Their wrapper dicts are independent.
    assert walker_a._wrappers is not walker_b._wrappers
