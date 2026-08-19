# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## Two Nodes

The pack ships exactly two ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res).

2. **`MiniMaxH3HarvestToConfig`** — harvest consumer. Parses a `harvest_json` STRING (produced by a **native** sampler pass, *not* by `MiniMaxH3SPEEDSampler`) into a human-readable calibration report.

> **Why this is two nodes, not more:** everything else (Schedule, SigmaHarvest) was deleted because it pretends to wire into generation but actually can't. The SPEED sampler takes raw widget values (`power_A`, `power_beta`, `transition_mode`, etc.), not a config object — so a node that emits `SpeedConfig` is a dead-end.

## Sigma Harvest: Must Run on a Native Sampler

The critical correctness constraint:

- **SPEED uses `(A, β)` to choose its transitions.** Harvest derives those same `(A, β)` from a residual power spectrum. Measuring the spectrum *inside* the SPEED sampler is circular and contaminates the residual with DCT-expand / boundary-align artifacts at each stage boundary.

- **Step indexing breaks across stages.** In `run_repeated_stage_calls`, `guider.sample()` is called *per stage* over a *sliced* sigma window. The callback's `step` is local to each stage, but the full sigma schedule is indexed globally — so after stage 1 the stamped sigma is wrong.

**Correct path:** run a **native Euler** full-res pass with the harvest callback active. Feed the resulting `harvest_json` into `MiniMaxH3HarvestToConfig` to read the fitted `power_A` / `power_beta`, then bake those into the SPEED sampler's widget defaults (or hardcode them in the workflow).

**Wrong path:** `MiniMaxH3SPEEDSampler` with `diagnostics="JSON"` — this hook instruments the wrong sampler and produces a mislabeled spectrum.

## Development Conventions

- SPEED calibrates on **Euler** only. Other samplers require re-deriving the kappa-alignment math.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Calibration happens offline; the baked defaults live in the node's widget defaults in `sampler_node.py`.
- No random configuration, no silent randomization in config paths.