"""Latent distinct-boundary lifecycle tests."""

import pytest
import torch

from speed_scripts.latent import Latent, RefLatent, LatentStage
from speed_scripts import h3_runtime as hr


def _kf(h, w):
    return {"latent": torch.rand(1, 1, 2, h, w)}


def test_latent_stage_flow_input_staged_inject_consumed():
    kf = _kf(8, 16)
    lat = Latent(kf)
    assert lat.stage == LatentStage.INPUT
    assert lat.current_hw == (8, 16)

    # downsample boundary — odd dims round UP to even
    lat.scale_to(3, 5)
    assert tuple(kf["latent"].shape[-2:]) == (4, 6)
    assert lat.stage == LatentStage.STAGED
    assert lat.current_hw == (4, 6)

    # same-size skip — no new tensor, stays staged
    before = kf["latent"]
    lat.scale_to(3, 5)
    assert kf["latent"] is before

    # pristine never degrades — second coarse resizes from pristine, not live
    lat2 = Latent(_kf(8, 16))
    orig = lat2.pristine.clone()
    lat2.scale_to(3, 5)
    lat2.scale_to(5, 9)
    assert tuple(lat2.pristine.shape[-2:]) == (8, 16)
    assert torch.equal(lat2.pristine, orig)

    # upsample/inject boundary
    lat2.restore()
    assert tuple(kf["latent"].shape[-2:]) != (0, 0)  # sanity
    assert lat2.stage == LatentStage.INJECT

    # consumed
    lat2.release()
    assert lat2.is_consumed
    with pytest.raises(RuntimeError):
        lat2.scale_to(8, 16)


def test_reflatent_never_scales():
    ref = _kf(8, 16)
    rl = RefLatent(ref)
    rl.scale_to(3, 5)
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)
    assert rl.stage == LatentStage.STAGED
    rl.restore()
    assert rl.stage == LatentStage.INJECT
    rl.release()
    assert rl.is_consumed


def test_h3_runtime_registry_sync_and_refs():
    hr._PRISTINE_STORE.clear()
    hr._LATENT_STORE.clear()

    kf1, kf2 = _kf(8, 16), _kf(8, 16)
    cond = {"minimax_keyframes": [kf1, kf2]}
    hr._rescale_cond_latents(cond, 5, 9, is_final_stage=False)
    assert len(hr._LATENT_STORE) == 2
    assert len(hr._PRISTINE_STORE) == 2
    assert all(tuple(k["latent"].shape[-2:]) == (6, 10) for k in cond["minimax_keyframes"])

    hr._rescale_cond_latents(cond, 8, 16, is_final_stage=True)
    assert len(hr._LATENT_STORE) == 0
    assert len(hr._PRISTINE_STORE) == 0
    assert all(tuple(k["latent"].shape[-2:]) == (8, 16) for k in cond["minimax_keyframes"])

    # refs are tracked but never rescaled
    hr._PRISTINE_STORE.clear()
    hr._LATENT_STORE.clear()
    ref = {"latent": torch.rand(1, 1, 2, 8, 16)}
    cond2 = {"minimax_keyframes": [_kf(8, 16)], "minimax_refs": [ref]}
    hr._rescale_cond_latents(cond2, 3, 5, is_final_stage=False)
    assert tuple(ref["latent"].shape[-2:]) == (8, 16)
    assert len(hr._LATENT_STORE) == 2  # one Latent + one RefLatent
    hr._rescale_cond_latents(cond2, 8, 16, is_final_stage=True)
    assert len(hr._LATENT_STORE) == 0
