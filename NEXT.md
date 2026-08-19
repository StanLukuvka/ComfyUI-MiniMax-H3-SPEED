# NEXT.md — ComfyUI-MiniMax-H3-SPEED (SPEED Sampler)

Last updated: 2026-08-18. Branch: `dev`. 132 tests passing.

## Status: MVP shipped + proven-load pass; branch hygiene + public face still open

The node pack installs and loads correctly in real ComfyUI v0.32.0 CPU mode
(verified: 852 node types registered, all 11 ours present, log banner fires).
All 4 workflows regenerated against the live registry schema and pass validation.

## Immediate work (highest priority first)

- [ ] **Port PR #3 (chflame163) into `dev`** — fixes `_active_av_shifts` for ComfyUI
      v0.33.1 (shifts on `model_sampling` + `transformer_options`, not `diffusion_model`).
      Dev currently has the broken probe at `h3_runtime.py:84-86`. This is issue #2.
      Add fallback-chain tests. Then push `dev`.
- [ ] **Decide `dev` → `main` promotion** — clean merge once PR #3 is in + README fixed.
      (main is the default branch; 30 strangers are cloning it and hitting old bugs.)
- [ ] **README overhaul** — fix the 6 lies: install URL `H3-SPEED.git` (self-referential
      alias), "required MiniMax-H3 plugin" (it's native core v0.32.0+),
      `transition_mode: "explicit"` (not a valid option anymore), shows deleted `nodes/`
      tree, test counts 61/74 (reality 132), ships `MiniMaxH3ImageToVideo` as external.
- [ ] **License decision** — `LICENSE.md` = MIT vs README PolyForm Noncommercial. Pick
      one, align both. User call.
- [ ] **GPU validation** — run `sigma_harvest.json` on Modal against real H3, pull fitted
      `power_A`/`power_beta`, update defaults. Then run t2v + coupled workflows end-to-end.
- [ ] **Repo hygiene** — move `verify_comfy_load.py` under `tools/`, decide if force-
      committed `.hermes/plans/` should be public, clear stale NEXT.md checkboxes.

## Done (for reference)

- [x] Flatten all nodes to flat root files (subpackage broke ComfyUI loading + collided
      with core `nodes` module)
- [x] Loud fail-fast logging (`h3_logging.py`, per-node registration + failure tracebacks)
- [x] Temporal DCT expansion (P5-003), MockNested removal (P5-005), temporal scale
      scheduling (P5-006) in `minimax_h3_speed/`
- [x] Regenerate all 4 workflows vs live v0.32.0 registry (was the real "not installed"
      cause — stale workflow JSONs, not the node install)
- [x] H3 power-spectrum defaults = 150.0 / 2.0 (estimates; replace after GPU harvest)

## Known deferrals (honest)

- [ ] P5-004 audio spectral expansion — no paper authority for audio `[B,C,2,T]`
- [ ] `temporal_scales` UI exposure — config+runtime only, no node widget yet

## Test counts (132 total)

- test_debug_nodes.py : 23
- test_spectral.py    : 7
- test_sampler.py     : 32
- test_config / runtime / harvest / integration / schedule_node : remaining
- Total: 132 passed

## Local test env

`/agent/comfyui-lab/` — Python 3.12 venv, CPU torch, ComfyUI v0.32.0, repo symlinked
into `custom_nodes/ComfyUI-MiniMaxH3-SPEED-Sampler`. Server up on `127.0.0.1:8188`.
Validation: `/agent/comfyui-lab/.venv/bin/python /agent/comfyui-lab/validate_workflows.py
workflows/*.json`.
