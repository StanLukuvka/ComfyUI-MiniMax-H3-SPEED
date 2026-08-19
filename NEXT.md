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

## Known deferrals (honest)

- [ ] P5-004 audio spectral expansion — no paper authority for audio `[B,C,2,T]`.
- [ ] `temporal_scales` UI exposure — config + runtime only, no node widget yet.
- [ ] GPU validation — fit `power_A` / `power_beta` from a real H3 harvest and
      bake the defaults (currently 150.0 / 2.0, paper estimates).
