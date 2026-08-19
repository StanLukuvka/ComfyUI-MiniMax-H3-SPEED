# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## Two Nodes

The pack ships exactly two ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res).

2. **`MiniMaxH3HarvestToConfig`** — harvest consumer. Parses a `harvest_json` STRING (produced by a **native** sampler pass, *not* by `MiniMaxH3SPEEDSampler`) into a human-readable calibration report.

> **Why this is two nodes, not more:** everything else (Schedule, SigmaHarvest) was deleted because it pretends to wire into generation but actually can't. The SPEED sampler takes raw widget values (`power_A`, `power_beta`, `transition_mode`, etc.), not a config object — so a node that emits `SpeedConfig` is a dead-end.

## Sigma Harvest: External Only

The pack does NOT harvest. The `harvest_json` STRING input on
`MiniMaxH3HarvestToConfig` is meant to come from a **native** Euler pass run
outside this pack (a workflow the user builds themselves, or another tool).
Nothing in this pack produces `harvest_json`, and the `diagnostics="JSON"` hook
that once instrumented the sampler has been deleted — it was circular (measured
the spectrum *inside* the SPEED sampler, contaminating the residual with
DCT artifacts) and broke step indexing across stages.

**Correct path:** run a **native** full-res Euler pass elsewhere with whatever
harvest tooling you prefer, then feed the resulting JSON into
`MiniMaxH3HarvestToConfig` to read the fitted `power_A` / `power_beta` and bake
those into the SPEED sampler's widget defaults (or hardcode them in the workflow).

**Wrong path:** trying to harvest from inside `MiniMaxH3SPEEDSampler` — there is
no hook for it anymore, and resurrecting one would re-introduce the circularity
and step-indexing bugs that got it deleted.

## Development Conventions

- SPEED calibrates on **Euler** only. Other samplers require re-deriving the kappa-alignment math.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Calibration happens offline; the baked defaults live in the node's widget defaults in `sampler_node.py`.
- No random configuration, no silent randomization in config paths.