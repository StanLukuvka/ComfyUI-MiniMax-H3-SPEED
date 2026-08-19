# AGENTS.md — ComfyUI-MiniMax-H3-SPEED (SPEED Sampler)

Agent-oriented session state. Branch: `dev`. Last sync: 2026-08-18.

## Reality Check (vs the rest of the repo's stale notes)

- **Branch:** `dev` is the working branch. `main` is 1 commit ahead (PR #1 merge) and
  ~21 commits behind. Default branch is `main` — strangers clone it and get the OLD
  code. `dev` is NOT auto-promoted.
- **Tests:** 132 passing (`.venv/bin/python -m pytest minimax_h3_speed/tests/ -q`).
- **Last real commit on `dev`:** `3b375ef` — regenerated all 4 workflows against the
  live ComfyUI v0.32.0 registry schema.

## Repo Layout (post-refactor)

- **All nodes are FLAT root-level files**, imported by `ROOT/__init__.py`:
  `sampler_node.py`, `sigma_harvest_node.py`, `schedule_node.py`,
  `harvest_to_config_node.py`, `inspect_node.py`, `power_spectrum_node.py`,
  `dct_lowpass_node.py`, `transition_math_node.py`, `spectral_expand_node.py`,
  `x0_fidelity_probe_node.py`, `av_reentry_oracle_node.py`. Shared helper: `common.py`.
- **Do NOT reintroduce the `nodes/` or `h3_speed_nodes/` subpackage.** It broke under
  ComfyUI's `importlib.util.spec_from_file_location` loading (root dir is not on
  sys.path), and `nodes` collides with ComfyUI core's OWN `nodes` module. Flat is the
  working pattern — match it for any new node.
- **Core logic:** `minimax_h3_speed/` — `h3_runtime.py` (sampling loop), `config.py`
  (SpeedConfig dataclass), `spectral.py` (DCT ops), `harvest.py` (HarvestCallback).
- **Node registration logging:** `h3_logging.py` — `get_logger(name)` returns a logger
  with handler at WARNING+, `[MINIMAX-H3-SPEED]` prefix, `propagate=False`. Use it in
  every node.
- **Proven-load smoke test:** `verify_comfy_load.py` — pre-loads a fake ComfyUI builtin
  `nodes` module then imports the pack; asserts all 11 nodes register.

## Critical Facts (verified against live ComfyUI v0.32.0)

- **`MiniMaxH3ImageToVideo`, `MiniMaxH3SigmaShift`, `MiniMaxH3ReferenceToVideo` are
  NATIVE ComfyUI core** (`comfy_extras/nodes_minimax_h3.py`). NOT a plugin. The README
  install section is WRONG about this. These register unconditionally (device chosen at
  runtime), so headless CPU load works.
- **Sigma shifts:** on ComfyUI v0.33.1 the active shifts live on
  `model.model_sampling` (ModelSamplingAV) + `transformer_options`, with stale
  `12.0`/`3.0` defaults on `diffusion_model`. Our `h3_runtime.py:84-86` reads
  `diffusion_model` — **broken on v0.33.1** (issue #2). PR #3 by chflame163 fixes this
  and targets `main`; NOT yet in `dev`.
- **Guider-based `sampler_node.py` is the ONLY working sampler.** There was a
  SAMPLER-type node in early history; it is gone. Do not reintroduce a SAMPLER-type node.

## Open Work (as of handover)

1. **Port PR #3 (chflame163) into `dev`** — sigma-shift fix for v0.33.1. Currently only
   on `main` via an open PR. Dev still broken on this.
2. **Decide `dev` → `main` promotion** — once #1 lands + README lies fixed.
3. **README overhaul** — 6 lies (install URL `H3-SPEED.git` alias, "required plugin"
   when it's native core, `transition_mode` "explicit" not a valid option, shows deleted
   `nodes/` tree, test counts 61/74 say reality 132, ships native node as external).
4. **License contradiction** — `LICENSE.md` = MIT, README says PolyForm Noncommercial.
   User decision needed.
5. **GPU validation + power-spectrum calibration** — `power_A=150.0`/`power_beta=2.0` are
   estimates; harvest never run on real H3 GPU. Workflows never executed (only schema-
   validated against live registry).
6. **Repo hygiene** — `verify_comfy_load.py` at root, force-committed `.hermes/plans/`
   public, `NEXT.md` has stale checkboxes.

## Deferred (honest)

- **P5-004 audio spectral expansion** — no paper authority (arXiv:2605.18736 covers
  spatial + temporal, not audio `[B,C,2,T]`). Deferred.
- **`temporal_scales` UI exposure** — in config + runtime, not in any node UI; waiting
  on H3 temporal geometry knowledge.

## How To Test Locally

ComfyUI lab at `/agent/comfyui-lab/` (Python 3.12 venv, CPU torch, ComfyUI v0.32.0,
our repo symlinked into `custom_nodes/`). Server still running on `127.0.0.1:8188`.
Validate workflows: `/agent/comfyui-lab/.venv/bin/python /agent/comfyui-lab/validate_workflows.py
workflows/*.json`.

Modal: `modal/comfyui.py` clones `dev` branch at image-build time. Rebuild:
`touch comfyui.py && modal serve comfyui.py`.
