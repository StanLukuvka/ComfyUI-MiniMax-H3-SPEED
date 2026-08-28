# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## Three Nodes

The pack ships exactly three ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** (Automatic) — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res). Picks `stages` (2-4), auto-computes the transition steps from `Tolerance (Delta)` + `noise_amplitude` + `noise_decay_exponent` via the power-spectrum threshold (`delta_custom` mode). Baked defaults: `Δ0.01 A7.394 β0.62` (balanced). Conservative `Δ0.005 A12.454 β0.819` available for sharper text.

2. **`MiniMaxH3SPEEDSamplerManual`** (Manual Step-Through) — same engine, explicit schedule. Up to four `(transition_goal, transition_resolution)` pairs; `goal == 0` or `resolution == 0` disables that stage. `ratio_mode steps` = goal is a step index, `ratio` = goal is a 0-1 fraction of the schedule. Used to copy paper schedules or test custom ladders.

3. **`MiniMaxH3HarvestToConfig`** (Sigma Harvest) — calibration tool. Runs one native full-res Euler pass (NOT the SPEED chain) with a fixed sigma schedule, captures `residual = x - denoised` per step, fits the radial DCT power spectrum `P = A·|ω|^-β`, and emits a flat `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r2`, `health`, `report`) to paste back into the Automatic node. Run it once when you change checkpoint, or when using Loras/addons that influence the model.

## Sigma Harvest: Native Euler only

`MiniMaxH3HarvestToConfig` wraps the **native** Euler sampler (`guider.sample()`), NOT `run_speed_pipeline`. It must run on a single full-res native Euler pass with a fixed sigma schedule.

**How to use it:** run the Harvest node at full-res with a fixed sigma schedule (28-32 steps `simple`), read the `calibration` JSON, paste `noise_amplitude` / `noise_decay_exponent` / `Tolerance (Delta)` into the Automatic node.

## Development Conventions

- SPEED calibrates on **Euler** only. Other samplers require re-deriving the kappa-alignment math.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Calibration happens offline; the baked defaults live in the node's widget defaults in `nodes/sampler_node.py` and `speed_scripts/config.py` (`SpeedConfig`).
- Latent boundaries: `speed_scripts/latent.py` (`Latent` / `RefLatent` / `LatentStage`) — pristine-only resize, even-round dims, id-keyed store synced with `_PRISTINE_STORE` in `h3_runtime.py`. `minimax_refs` are `RefLatent` (never scaled).
- No random configuration, no silent randomization in config paths.
- Tests: `speed_scripts/tests/` — `54 passed, 5 skipped` (5 skips are `speed_lab` sibling deselected).
