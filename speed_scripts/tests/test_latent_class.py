"""Latent holder/walker lifecycle contracts."""

import torch

from speed_scripts.latent_class import LatentClass, LatentStage, LatentWalker


def _holder(h=8, w=16):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def _guider(positive, negative=()):
    return type(
        "Guider",
        (),
        {"original_conds": {"positive": list(positive), "negative": list(negative)}},
    )()


def test_latent_class_always_resizes_from_pristine_and_restores():
    keyframe = _holder()
    pristine = keyframe["latent"].clone()
    latent = LatentClass(keyframe)

    latent.downscale(3, 5)
    assert tuple(keyframe["latent"].shape[-2:]) == (4, 6)
    latent.downscale(5, 9)
    assert tuple(keyframe["latent"].shape[-2:]) == (6, 10)
    assert tuple(latent.pristine.shape[-2:]) == (8, 16)
    assert latent.stage == LatentStage.STAGED

    latent.upscale_to_inject()
    assert tuple(keyframe["latent"].shape[-2:]) == (8, 16)
    assert torch.equal(keyframe["latent"], pristine)
    latent.release()
    assert latent.is_consumed


def test_walker_handles_all_conds_keyframes_and_refs_then_releases():
    pos_a, pos_b, neg = _holder(), _holder(), _holder()
    ref = _holder()
    positive = {"minimax_keyframes": [pos_a, pos_b], "minimax_refs": [ref]}
    negative = {"minimax_keyframes": [neg], "minimax_refs": []}
    walker = LatentWalker(_guider([positive], [negative]))

    walker.apply_stage(3, 5)
    for keyframe in (pos_a, pos_b, neg):
        assert tuple(keyframe["latent"].shape[-2:]) == (4, 6)
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)

    walker.apply_final()
    for keyframe in (pos_a, pos_b, neg, ref):
        assert tuple(keyframe["latent"].shape[-2:]) == (8, 16)
    assert not walker._wrappers


def test_walkers_are_generation_local():
    a, b = _holder(), _holder()
    walker_a = LatentWalker(_guider([{"minimax_keyframes": [a]}]))
    walker_b = LatentWalker(_guider([{"minimax_keyframes": [b]}]))
    walker_a.apply_stage(3, 5)
    assert tuple(a["latent"].shape[-2:]) == (4, 6)
    assert tuple(b["latent"].shape[-2:]) == (8, 16)
    assert walker_a._wrappers is not walker_b._wrappers
