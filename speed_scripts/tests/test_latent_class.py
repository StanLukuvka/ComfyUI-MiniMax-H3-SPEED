"""LatentWalker: I2V keyframe resizing from pristine latents and restore."""

import torch

from speed_scripts.latent_class import LatentWalker


def _holder(h=8, w=16):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def _guider(positive, negative=()):
    return type(
        "Guider",
        (),
        {"original_conds": {"positive": list(positive), "negative": list(negative)}},
    )()


def test_walker_resizes_from_pristine_and_restores():
    keyframe = _holder()
    pristine = keyframe["latent"].clone()
    walker = LatentWalker(_guider([{"minimax_keyframes": [keyframe]}]))

    walker.apply_stage(3, 5)
    assert tuple(keyframe["latent"].shape[-2:]) == (4, 6)
    walker.apply_stage(5, 9)
    assert tuple(keyframe["latent"].shape[-2:]) == (6, 10)

    walker.apply_final()
    assert torch.equal(keyframe["latent"], pristine)

    # The restore releases the keyframes; the walker no longer resizes them.
    walker.apply_stage(3, 5)
    assert torch.equal(keyframe["latent"], pristine)


def test_walker_handles_all_condition_groups_and_leaves_refs_untouched():
    pos_a, pos_b, neg = _holder(), _holder(), _holder()
    ref = _holder()
    ref_identity = ref["latent"]
    positive = {"minimax_keyframes": [pos_a, pos_b], "minimax_refs": [ref]}
    negative = {"minimax_keyframes": [neg], "minimax_refs": []}
    walker = LatentWalker(_guider([positive], [negative]))

    walker.apply_stage(3, 5)
    for keyframe in (pos_a, pos_b, neg):
        assert tuple(keyframe["latent"].shape[-2:]) == (4, 6)
    assert ref["latent"] is ref_identity

    walker.apply_final()
    for keyframe in (pos_a, pos_b, neg, ref):
        assert tuple(keyframe["latent"].shape[-2:]) == (8, 16)
