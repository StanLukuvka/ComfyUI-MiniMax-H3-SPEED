# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## V2 Release Contract

V2 is the major-release boundary for the multi-sampler/runtime rewrite. The public pack still ships exactly three nodes, but V2 expands the supported sampler surface and replaces most of the V1 execution/planning internals.

Release-level invariants:

- Existing Automatic and Manual calls that omit `sampler_name` continue to run Euler.
- Existing Automatic/Manual inputs keep their order; sampler selection is appended rather than inserted between old widgets.
- The three public node IDs remain stable.
- Euler remains the reference path for baked calibration and benchmark evidence.
- RES Multistep is deterministic, run-scoped, and reset-only at resolution boundaries. There is no history-mode widget.
- `direct_coarse` remains the default noise policy. `coupled_full_grid` remains available as a deterministic full-grid coupling/ablation path, not as a claimed quality preset.
- Internal `speed_scripts` APIs may break from V1 where required by the planner/runtime split; see `CHANGELOG.md` for migration details.

## Three Nodes

The pack ships exactly three ComfyUI nodes:

1. **`MiniMaxH3SPEEDSampler`** (Automatic) — the generator. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 by running a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res). Picks `stages` (2-4), auto-computes the transition steps from `Tolerance (Delta)` + `noise_amplitude` + `noise_decay_exponent` via the power-spectrum threshold (`delta_custom` mode). Baked defaults: `Δ0.005 A12.105 β0.773` (conservative, 0.5%). Balanced `Δ0.01 A12.436 β0.786` runs faster at near parity.

2. **`MiniMaxH3SPEEDSamplerManual`** (Manual Step-Through) — same engine, explicit schedule. Up to four `(transition_goal, transition_resolution)` pairs; `goal == 0` or `resolution == 0` disables that stage. `resolution` is the stage scale in both modes. `ratio_mode steps` = goal is a step index (whole numbers only), `ratio` = goal is a 0-1 fraction of the schedule; the boundary is placed at `round(goal * total_steps)`. Used to copy paper schedules or test custom ladders.

3. **`MiniMaxH3HarvestToConfig`** (Sigma Harvest) — calibration tool. Runs one native full-resolution pass with the selected sampler (NOT the SPEED chain), computes `residual = x - denoised` per step, immediately reduces each residual to a CPU radial DCT power profile, fits `P = A·|ω|^-β`, and emits a flat sampler-specific `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r2`, `health`, `report`) to paste back into the matching Automatic configuration. Run it when you change checkpoint, sampler, LoRA/addons, or the sigma schedule.

## Sigma Harvest: Selected native sampler

`MiniMaxH3HarvestToConfig` wraps the selected **native** Comfy sampler (`guider.sample()`), NOT `run_speed_pipeline`. It runs one full-resolution native pass with a fixed sigma schedule.

**How to use it:** run the Harvest node at full-res with the sampler you intend to use in SPEED, using the same sigma scheduler and step count you intend to run. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler. Read the sampler-specific `calibration` JSON and paste its `noise_amplitude` / `noise_decay_exponent` / `Tolerance (Delta)` into the matching Automatic run. For `res_multistep`, Harvest uses native Comfy `res_multistep`; SPEED generation currently uses the repository's stateful RES adapter.

## Calibration

Baked defaults and current evidence are Euler-derived. Changing the checkpoint, LoRA/addons, sampler, or materially changing the sigma schedule is a reason to re-harvest.

## Sampler architecture

Stateless samplers use native Comfy sampler objects. `res_multistep` uses a run-scoped stateful adapter that clears all previous-step RES history at every SPEED stage boundary. There is no history-mode widget or switch. The global SPEED scheduler remains sampler-agnostic. The supported RES adapter is deterministic and non-ancestral only: no SDE and no CFG++.

## Noise policies

- **`direct_coarse`** — default. Starts on coarse Gaussian noise and fills newly exposed frequency bands from deterministic transition-seeded Gaussian noise.
- **`coupled_full_grid`** — builds one seeded full-resolution Gaussian field, transforms it once to spectral coefficients, and reuses the relevant coefficient bands at each resolution transition. Its purpose is deterministic coupling to one full-grid realization and parity/ablation work. V2 does not claim it is generally sharper or higher quality than `direct_coarse`.

## Development Conventions

- SPEED's baked defaults and current evidence are Euler-derived. Re-harvest when changing checkpoint, sampler, LoRA/addons, or materially changing the sigma schedule; do not claim parity for unmeasured samplers.
- All supported sampler paths are deterministic and non-ancestral. This release does not add ancestral, SDE, or CFG++ variants.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SIGMAS`, `LATENT`).
- Automatic configs do not store placeholder transition indices or latent dimensions. `delta_custom` boundaries are computed from the live sigma schedule and live H3 latent geometry at runtime; `transition_steps` is only meaningful in explicit/manual mode.
- Stage geometry, transition-threshold math, Automatic config construction, and Manual schedule normalization live together in `speed_scripts/planning.py`. `speed_scripts/automatic_config.py` is only a compatibility re-export for older imports.
- `speed_scripts/h3_runtime.py` owns execution rather than planning: H3 latent validation, preview timeline, spectral/audio boundary application, sampler hooks, stage calls, and output assembly. Coupled full-grid noise is transformed to spectral coefficients once per generation and reused at every transition.
- I2V latent lifecycle lives in `speed_scripts/latent_class.py` as one `LatentWalker`. It snapshots only keyframe latents, always resizes from pristine full resolution, restores them before the final stage and on failure, and never wraps or resizes `minimax_refs`. The walker is local to one `run_speed_pipeline()` call; it is not stored on the guider.
- Sigma Harvest reduces each captured full-resolution residual to its CPU radial power profile inside the sampler callback; it does not retain a run's full residual tensors.
- No random configuration, no silent randomization in config paths.
- Tests: `speed_scripts/tests/` — run with the repo venv (`.venv/bin/python -m pytest speed_scripts/tests/ -q`); the repo has no CI workflows for dev PRs, so the local suite is the gate.
