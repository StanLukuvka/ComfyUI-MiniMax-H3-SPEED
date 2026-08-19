# NEXT.md — ComfyUI-MiniMax-H3-SPEED (SPEED Sampler)

Last updated: 2026-08-19. Branch: `main` (== `dev`).

## Status: MVP shipped, docs + license aligned, pack is clean

The node pack installs and loads in ComfyUI (verified: 2 nodes registered,
`MiniMax H3 SPEED — Sampler` + `MiniMax H3 SPEED — Harvest to Config`).
42 tests pass. License is PolyForm Noncommercial 1.0.0 (aligned across
LICENSE.md, README, pyproject). README is truthful and per-line verified.

## What actually shipped

- `MiniMaxH3SPEEDSampler` — runs the multi-stage SPEED chain (Euler) on the
  MiniMax-H3 latent. Widgets: `scales`, `transition_steps`, `transition_mode`
  (manual_step / manual_sigma / delta_custom), `noise_policy` (direct_coarse /
  coupled_full_grid), `delta`, `power_A`, `power_beta`, `seed_offset`,
  `explicit_preset` (Half then Full / Quarter then Half / Aggressive / Three
  Quarter), `full_latent_h/w`, `integration_steps`, `clip_ratio`.
- `MiniMaxH3HarvestToConfig` — parses a `harvest_json` STRING (produced by a
  **native** sampler pass elsewhere) into a human-readable calibration report.
- `minimax_h3_speed/` — `config.py`, `flow.py` (networkx stage graph),
  `h3_runtime.py` (sigma-shift fallback chain), `harvest.py` (numpy power
  spectrum + power-law fit), `spectral.py` (DCT expansion).
- `workflows/video_minimax_h3_SPEED.json` — one reference workflow.
- 42 tests across `test_dct / test_flow / test_sampler / test_spectral`.

## Dead-end nodes (deleted, do NOT resurrect)

Schedule, SigmaHarvest, Oracle, inspect, and the in-SPEED harvest hook were all
removed. They pretended to wire into generation but couldn't — the SPEED sampler
takes raw widget values, not a config object, so a node emitting `SpeedConfig`
was a dead-end. Similarly the in-SPEED harvest hook measured the spectrum
*inside* the sampler (circular contamination) and broke step indexing across
stages, so it was cut. Harvest now runs only on a native Euler pass.

## Done

- [x] Flatten nodes to flat root files (the `nodes/` subpackage collided with
      ComfyUI's internal `nodes/` namespace and broke loading).
- [x] PolyForm license decision (LICENSE.md was briefly MIT in one commit;
      history-rewritten so PolyForm is the only license ever present).
- [x] README overhaul — verified truthful, no lies.
- [x] Dependency declarations fixed: numpy + networkx now in pyproject/requirements.
- [x] dev → main promotion; dev tracks main.

## CRITICAL: Sigma Harvest design rule (do NOT violate)

The harvest node (`MiniMaxH3HarvestToConfig`) is a **native Euler sampler wrapper**,
NOT a SPEED-chain wrapper. The SPEED sampler splices/re-aligns sigmas at every
stage boundary, which corrupts per-step sigma labels — you CANNOT harvest a
meaningful spectrum mid-SPEED-chain. The harvester runs ONE full-res native
Euler pass with a FIXED sigma schedule, captures residual = x - x0 at each step,
fits P = A*|ω|^(-β), emits `harvest_json`. Same inputs as any sampler node
(noise/guider/sigmas/latent_image). If you ever see the harvest node take a
`harvest_json` STRING input instead of native sampler inputs, it has regressed.

## CRITICAL: audio sigma-shift lookup (garbled sound regression)

`_active_av_shifts` in h3_runtime.py picks the H3 sigma shifts. Priority MUST be:
(1) `transformer_options["minimax_h3_sigma_shift_*"]`, (2) `model.sigma_shift_*`
(12.0 / 3.0 — AUTHORITATIVE), (3) `diffusion_model.sigma_shift_*`, (4) LAST RESORT
`model_sampling.shift` (generic ComfyUI flow shift, often 1.0 — WRONG for H3).

Commit 84e61ba ("Fix MiniMax-H3 sigma shift lookup") inverted this: it put
`model_sampling.shift` at HIGHER priority than `sigma_shift_video`. That collapsed
`audio_scale` from 4.0 to ~0.333, rescaling every audio transition ~12x wrong →
garbled/awful sound. The regression test `test_av_shifts_ignore_generic_model_sampling_shift`
locks the correct priority. NEVER let generic `model_sampling.shift` shadow the
H3-specific `sigma_shift_video`/`audio`.

## CRITICAL: SPEED progress bar must stay ON

`run_repeated_stage_calls` defaults `disable_pbar=False` (bar VISIBLE). The SPEED
sampler node passes `disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED`. Do NOT
change the runtime default back to `True` — it hides the bar for every run and is
easy to miss.

- [ ] P5-004 audio spectral expansion — no paper authority for audio `[B,C,2,T]`.
- [ ] `temporal_scales` UI exposure — config + runtime only, no node widget yet.
- [ ] GPU validation — fit `power_A` / `power_beta` from a real H3 harvest and
      bake the defaults (currently 150.0 / 2.0, paper estimates).
