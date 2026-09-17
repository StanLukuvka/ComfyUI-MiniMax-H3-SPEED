# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## Three Nodes

The pack ships exactly three ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** (Automatic) — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res). Picks `stages` (2-4), auto-computes the transition steps from `Tolerance (Delta)` + `noise_amplitude` + `noise_decay_exponent` via the power-spectrum threshold (`delta_custom` mode). Baked defaults: `Δ0.005 A12.105 β0.773` (conservative, 0.5%). Balanced `Δ0.01 A12.436 β0.786` runs faster at near parity.

2. **`MiniMaxH3SPEEDSamplerManual`** (Manual Step-Through) — same engine, explicit schedule. Up to four `(transition_goal, transition_resolution)` pairs; `goal == 0` or `resolution == 0` disables that stage. `resolution` is the stage scale in both modes. `ratio_mode steps` = goal is a step index (whole numbers only), `ratio` = goal is a 0-1 fraction of the schedule; the boundary is placed at `round(goal * total_steps)`. Used to copy paper schedules or test custom ladders.

3. **`MiniMaxH3HarvestToConfig`** (Sigma Harvest) — calibration tool. Runs one native full-resolution pass with the selected sampler (NOT the SPEED chain), captures `residual = x - denoised` per step, fits the radial DCT power spectrum `P = A·|ω|^-β`, and emits a flat sampler-specific `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r2`, `health`, `report`) to paste back into the matching Automatic configuration. Run it when you change checkpoint, sampler, LoRA/addons, or the sigma schedule.

## Sigma Harvest: Selected native sampler

`MiniMaxH3HarvestToConfig` wraps the selected **native** Comfy sampler (`guider.sample()`), NOT `run_speed_pipeline`. It runs one full-resolution native pass with a fixed sigma schedule.

**How to use it:** run the Harvest node at full-res with the sampler you intend to use in SPEED, using the same sigma scheduler and step count you intend to run. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler. Read the sampler-specific `calibration` JSON and paste its `noise_amplitude` / `noise_decay_exponent` / `Tolerance (Delta)` into the matching Automatic run. For `res_multistep`, Harvest uses native Comfy `res_multistep`; SPEED generation uses the repository's stateful RES adapter.

## Calibration

Baked defaults and current evidence are Euler-derived. Changing the checkpoint, LoRA/addons, sampler, or materially changing the sigma schedule is a reason to re-harvest.

## Sampler architecture

Stateless samplers use native Comfy sampler objects. `res_multistep` uses a run-scoped stateful adapter that clears all previous-step RES history at every SPEED stage boundary (reset behavior). There is no history-mode widget or switch. The global SPEED scheduler remains sampler-agnostic. The supported RES adapter is deterministic and non-ancestral only: no SDE and no CFG++.

## Development Conventions

- SPEED's baked defaults and current evidence are Euler-derived. Re-harvest when changing checkpoint, sampler, LoRA/addons, or materially changing the sigma schedule; do not claim parity for unmeasured samplers.
- All supported sampler paths are deterministic and non-ancestral. This release does not add ancestral, SDE, or CFG++ variants.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Calibration happens offline; the baked defaults live in the node's widget defaults in `nodes/sampler_node.py` (`speed_scripts/config.py` holds the SpeedConfig dataclass defaults, which the node path always overrides explicitly). The stage ladder + config assembly is centralized in `speed_scripts/automatic_config.py` (`build_automatic_speed_config`).
- Latent lifecycle: `speed_scripts/latent_class.py` (`LatentClass` / `LatentWalker`, plus `LatentStage`) — the walker snapshots pristine full-res cond/ref latents once per run, `apply_stage(h, w)` resizes keyframes from pristine for each coarse stage (even-round dims, never from degraded tensors), `apply_final()` restores full res. `minimax_refs` are never scaled (their row allocation is locked to full res by the model). The walker is stashed on the guider per run (`_LW_ATTR` in `h3_runtime.py`) and dropped at run end.
- No random configuration, no silent randomization in config paths.
- Tests: `speed_scripts/tests/` — run with the repo venv (`.venv/bin/python -m pytest speed_scripts/tests/ -q`); the repo has no CI workflows for dev PRs, so the local suite is the gate.
