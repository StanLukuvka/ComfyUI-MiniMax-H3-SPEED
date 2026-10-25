# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

This file is the source of truth for how this pack is intended to be used, developed, and maintained.

## V2 Release Contract

V2 is the major-release boundary for the multi-sampler/runtime rewrite. The public pack still ships exactly three nodes, but V2 expands the supported sampler surface and replaces most of the V1 execution/planning internals.

Release-level invariants:

- The three public node IDs are `SPEEDSampler`, `SPEEDSamplerManual` and `SPEEDSigmaHarvest`. The old IDs `MiniMaxH3SPEEDSampler`, `MiniMaxH3SPEEDSamplerManual` and `MiniMaxH3HarvestToConfig` stay registered as `DEPRECATED` subclasses, plus node replacements, so saved workflows still load. Keep both sets stable.
- Euler remains the reference path for baked calibration and benchmark evidence.
- All three nodes take a standard ComfyUI `SAMPLER` input (see Sampler architecture).
- All three nodes take a required `model_type` switch (`h3_video`, `flow_image`) with no default anywhere (widget, node signature, runtime, Harvest helpers); see Model families. A missing value fails validation; a mismatched one raises.
- Nothing model-specific is a default. Automatic's `noise_amplitude` / `noise_decay_exponent` start at 0 (unset, raises), and `SpeedConfig` has no baked A/beta/delta.
- `direct_coarse` remains the default noise policy. `coupled_full_grid` remains available because its removal has not been shown to be safe; V2 makes no claim that it improves quality.
- Internal `speed_scripts` APIs may break from V1 where required by the planner/runtime split; see `CHANGELOG.md` for migration details.

## Three Nodes

The pack ships exactly three ComfyUI nodes:

1. **`SPEEDSampler`** (Automatic) — the generator. It uses the same `noise` / `guider` / `sampler` / `sigmas` / `latent_image` interface as `SamplerCustomAdvanced` and runs a multi-stage progressive-resolution diffusion pass (low-res first, boundary-align, then full-res). A basic all-in-one KSampler workflow must first be split into the advanced sampling components. Picks `stages` (2-4), auto-computes the transition steps from `Tolerance (Delta)` + `noise_amplitude` + `noise_decay_exponent` via the power-spectrum threshold (`delta_custom` mode). No A/beta defaults; H3 Euler reference values: `Δ0.005 A12.105 β0.773` (conservative, 0.5%). Balanced `Δ0.01 A12.436 β0.786` runs faster at near parity.

2. **`SPEEDSamplerManual`** (Manual Step-Through) — same engine, explicit schedule. Up to four `(transition_goal, transition_resolution)` pairs; `goal == 0` or `resolution == 0` disables that stage. `resolution` is the stage scale in both modes. `ratio_mode steps` = goal is a step index (whole numbers only), `ratio` = goal is a 0-1 fraction of the schedule; the boundary is placed at `round(goal * total_steps)`. Used to copy paper schedules or test custom ladders.

3. **`SPEEDSigmaHarvest`** (Sigma Harvest) — calibration tool. Runs one native full-resolution pass with the connected sampler (NOT the SPEED chain), computes `residual = x - denoised` per step, immediately reduces each residual to a CPU radial DCT power profile, fits `P = A·|ω|^-β`, and emits a flat sampler-specific `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r2`, `health`, `report`) to paste back into the matching Automatic configuration. Run it when you change checkpoint, sampler, LoRA/addons, or the sigma schedule.

## Sigma Harvest: connected native sampler

`SPEEDSigmaHarvest` runs the connected **native** Comfy sampler (`guider.sample()`), NOT `run_speed_pipeline`. It runs one full-resolution native pass with a fixed sigma schedule.

**How to use it:** run the Harvest node at full-res with the sampler you intend to use in SPEED, using the same sigma scheduler and step count you intend to run. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler. Read the sampler-specific `calibration` JSON and paste its `noise_amplitude` / `noise_decay_exponent` / `Tolerance (Delta)` into the matching Automatic run. Harvest and SPEED run the same sampler object, so the calibration matches the sampler that generates.

## Calibration

Baked defaults and current evidence are Euler-derived. Changing the checkpoint, LoRA/addons, sampler, or materially changing the sigma schedule is a reason to re-harvest.

## Sampler architecture

Any ComfyUI SAMPLER is accepted; there is no allowlist and no per-sampler adapter. SPEED passes the connected `SAMPLER` object to `guider.sample()` once per stage. ComfyUI keeps a multistep sampler's history local to one call, so the history resets at every resolution change without any adapter. Ancestral and SDE samplers draw their per-step noise from the run seed in each call. The global SPEED scheduler is sampler-agnostic. Tested results per sampler are in the README.

## Model families

`speed_scripts/families/` is a package with one class per `model_type`: the registry and public re-exports live in `families/__init__.py` (`MODEL_TYPES`, `get_family`, `check_model_type`, `prepare_latent`, `main_stream`), `families/h3_video.py` holds `H3VideoFamily` with its unpack/pack, sigma-shift and audio-carry helpers, and `families/flow_image.py` holds `FlowImageFamily` with `require_flow_sampling`. The runtime is family-agnostic. A family splits the latent into the spatial stream SPEED resizes (always `[B,C,T,H,W]` inside the runtime) and an optional extra stream, reads a stage's output back into sampler state (`to_internal`), and carries the extra stream across a resolution change (`carry_extra`).

- `h3_video` — nested video/audio latent; audio carried on its own sigma clock with the H3 shifts. Output is bit-identical to the pre-families runtime.
- `flow_image` — plain image latent from a CONST (flow) model. `to_internal` undoes the latent format's `process_latent_in` (Krea 2 uses Wan 2.1's mean/std), because ComfyUI returns `process_latent_out(x / (1 - sigma))`.

The model type is an explicit switch, never auto-detected; a mismatch raises with the setting to use. Add a model by adding a family, not by branching in the runtime.

## Noise policies

- **`direct_coarse`** — default. Starts on coarse Gaussian noise and fills newly exposed frequency bands from deterministic transition-seeded Gaussian noise.
- **`coupled_full_grid`** — builds one seeded full-resolution Gaussian field, transforms it once to spectral coefficients, and reuses the relevant coefficient bands at each resolution transition. It remains available because there is not enough evidence to confidently remove it. V2 does not claim it improves quality over `direct_coarse`.

## Development Conventions

- SPEED's baked defaults and current evidence are Euler-derived. Re-harvest when changing checkpoint, sampler, LoRA/addons, or materially changing the sigma schedule; do not claim parity for unmeasured samplers.
- Document a sampler as recommended only after a measured run; known-bad samplers go in the README.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SAMPLER`, `SIGMAS`, `LATENT`).
- Automatic configs do not store placeholder transition indices or latent dimensions. `delta_custom` boundaries are computed from the live sigma schedule and live latent geometry at runtime; `transition_steps` is only meaningful in explicit/manual mode.
- Stage geometry, transition-threshold math, Automatic config construction, and Manual schedule normalization live together in `speed_scripts/planning.py`.
- `speed_scripts/runtime.py` owns execution rather than planning: preview timeline, spectral boundary application, stage calls, and output assembly. Latent layout and extra streams belong to the family. Coupled full-grid noise is transformed to spectral coefficients once per generation and reused at every transition.
- I2V latent lifecycle lives in `speed_scripts/latent_class.py` as one `LatentWalker`. It snapshots only keyframe latents, always resizes from pristine full resolution, restores them before the final stage and on failure, and never wraps or resizes `minimax_refs`. The walker is local to one `run_speed_pipeline()` call; it is not stored on the guider.
- Sigma Harvest reduces each captured full-resolution residual to its CPU radial power profile inside the sampler callback; it does not retain a run's full residual tensors.
- No random configuration, no silent randomization in config paths.
- Tests: `speed_scripts/tests/` — run with the repo venv (`.venv/bin/python -m pytest speed_scripts/tests/ -q`); the repo has no CI workflows for dev PRs, so the local suite is the gate.
