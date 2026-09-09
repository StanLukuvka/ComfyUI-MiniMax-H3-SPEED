# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## Three Nodes

The pack ships exactly three ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** (Automatic) — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res). Picks `stages` (2-4), auto-computes the transition steps from `Tolerance (Delta)` + `noise_amplitude` + `noise_decay_exponent` via the power-spectrum threshold (`delta_custom` mode). Baked defaults: `Δ0.005 A12.105 β0.773` (conservative, 0.5%). Balanced `Δ0.01 A12.436 β0.786` runs faster at near parity.

2. **`MiniMaxH3SPEEDSamplerManual`** (Manual Step-Through) — same engine, explicit schedule. Up to four `(transition_goal, transition_resolution)` pairs; `goal == 0` or `resolution == 0` disables that stage. `resolution` is the stage scale in both modes. `ratio_mode steps` = goal is a step index (whole numbers only), `ratio` = goal is a 0-1 fraction of the schedule; the boundary is placed at `round(goal * total_steps)`. Used to copy paper schedules or test custom ladders.

3. **`MiniMaxH3HarvestToConfig`** (Sigma Harvest) — calibration tool. Runs one native full-res Euler pass (NOT the SPEED chain) with a fixed sigma schedule, captures `residual = x - denoised` per step, fits the radial DCT power spectrum `P = A·|ω|^-β`, and emits a flat `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r2`, `health`, `report`) to paste back into the Automatic node. Run it once when you change checkpoint, or when using Loras/addons that influence the model.

## Sigma Harvest: Native Euler only

`MiniMaxH3HarvestToConfig` wraps the **native** Euler sampler (`guider.sample()`), NOT `run_speed_pipeline`. It must run on a single full-res native Euler pass with a fixed sigma schedule.

**How to use it:** run the Harvest node at full-res with a fixed sigma schedule (28-32 steps `simple`), read the `calibration` JSON, paste `noise_amplitude` / `noise_decay_exponent` / `Tolerance (Delta)` into the Automatic node.

## Development Conventions

- SPEED calibrates on **Euler** only. Other samplers require re-deriving the kappa-alignment math.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Calibration happens offline; the baked defaults live in the node's widget defaults in `nodes/sampler_node.py` (`speed_scripts/config.py` holds the SpeedConfig dataclass defaults, which the node path always overrides explicitly). The stage ladder + config assembly is centralized in `speed_scripts/automatic_config.py` (`build_automatic_speed_config`).
- Latent lifecycle: `speed_scripts/latent_class.py` (`LatentClass` / `LatentWalker`, plus `LatentStage`) — the walker snapshots pristine full-res cond/ref latents once per run, `apply_stage(h, w)` resizes keyframes from pristine for each coarse stage (even-round dims, never from degraded tensors), `apply_final()` restores full res. `minimax_refs` are never scaled (their row allocation is locked to full res by the model). The walker is stashed on the guider per run (`_LW_ATTR` in `h3_runtime.py`) and dropped at run end.
- No random configuration, no silent randomization in config paths.
- Tests: `speed_scripts/tests/` — run with the repo venv (`.venv/bin/python -m pytest speed_scripts/tests/ -q`); the repo has no CI workflows for dev PRs, so the local suite is the gate.

