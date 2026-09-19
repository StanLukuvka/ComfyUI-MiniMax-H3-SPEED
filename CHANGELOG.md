# Changelog

This project uses major release notes for changes that materially alter the public sampler surface or the internal SPEED execution model.

## 2.0.0 — Unreleased

V2 is a major rewrite of the MiniMax-H3 SPEED node pack. The three public node identities remain, but most of the execution, planning, sampler, calibration, and test architecture has been replaced or expanded.

### User-facing changes

- **Five supported samplers.** Automatic, Manual, and Sigma Harvest now share the same sampler selector: Euler, Heun, DPM2 (`dpm_2`), Exp Heun 2 X0 (`exp_heun_2_x0`), and RES Multistep (`res_multistep`).
- **Euler remains the default and reference path.** Existing Automatic/Manual calls that omit `sampler_name` still execute Euler.
- **Stateful RES support.** SPEED uses a run-scoped deterministic RES Multistep adapter. RES history is reset at every resolution transition so history from one latent grid is never reused on another grid.
- **Sampler-aware Sigma Harvest.** Harvest now runs the selected native full-resolution sampler and emits sampler identity with the calibration. A calibration should be harvested with the same sampler, checkpoint, LoRA/addons, sigma scheduler, and step count intended for the SPEED run.
- **Automatic planning uses live runtime geometry.** Transition placement is computed from the live sigma schedule and the actual H3 latent dimensions rather than cached dimensions stored in configuration.
- **Manual scheduling is normalized through the shared planner.** Explicit step and ratio modes use the same validation and runtime configuration path as Automatic.
- **I2V conditioning lifecycle is generation-local.** Keyframe latents are snapshotted from pristine full resolution, resized for coarse stages, restored before the final stage and on failure, and never accumulated through repeated interpolation. `minimax_refs` remain untouched.
- **Continuous progress/preview timeline.** Multi-stage sampling reports progress as one run-wide denoising sequence instead of independent stage-local progress.
- **Output contracts are pinned.** Both normal and denoised outputs preserve the nested MiniMax-H3 video/audio latent structure.

### Noise policies

Both existing policies remain available:

- `direct_coarse` remains the default. It starts from coarse Gaussian noise and fills newly exposed frequency bands with deterministic transition-seeded Gaussian noise.
- `coupled_full_grid` derives stage noise from one seeded full-resolution Gaussian field and reuses its spectral coefficients as resolution grows. V2 computes that full-grid transform once per generation and reuses it at every transition.

`coupled_full_grid` is retained because it gives a well-defined deterministic coupling to one full-resolution noise realization and is useful for parity/ablation work. It should **not** currently be read as a proven quality or sharpness mode; no V2 evidence establishes a general quality advantage over `direct_coarse`.

### Sigma Harvest changes

- Harvest remains a **native full-resolution pass**, not a SPEED pass.
- Calibration is explicitly identified as an empirical residual fit of `x - denoised`, not the clean-data power spectrum from the SPEED paper.
- Residuals are reduced to small CPU radial DCT power profiles inside the sampler callback rather than retaining every full-resolution residual tensor until the end of the run.
- JSON error output is structured and safely encoded.
- The diagnostic latent remains available as the second output.

### Internal architecture

- Added `speed_scripts/planning.py` as the owner of stage geometry, threshold math, Automatic config construction, and Manual schedule normalization.
- `speed_scripts/h3_runtime.py` now focuses on execution: latent validation, stage calls, boundary conversion, audio handling, spectral expansion, sampler hooks, preview/progress, and output assembly.
- Added `speed_scripts/sampler_support.py` for the supported sampler surface and run-scoped sampler handles.
- Added `speed_scripts/res_multistep_adapter.py` for deterministic stateful RES execution.
- Consolidated I2V handling into one generation-local `LatentWalker`.
- Removed the old `nodes_common.py` helper layer.
- `automatic_config.py` is now only a compatibility re-export.
- Configuration no longer stores placeholder Automatic transition indices or cached full latent H/W.
- Coupled full-grid spectral coefficients are computed once per generation instead of once per transition.

### Compatibility and migration

Ordinary ComfyUI workflows are intended to remain compatible:

- The three public node IDs remain `MiniMaxH3SPEEDSampler`, `MiniMaxH3SPEEDSamplerManual`, and `MiniMaxH3HarvestToConfig`.
- Existing Automatic and Manual input order is preserved; `sampler_name` is appended and defaults to Euler.
- Legacy Tolerance aliases and old Automatic preset aliases remain accepted by the node implementation.
- Committed example workflows explicitly select Euler.

The internal Python API is **not** V1-compatible in every detail:

- `SpeedConfig.full_latent_h` and `SpeedConfig.full_latent_w` were removed.
- A `SpeedConfig` must represent at least two resolution stages.
- `build_automatic_speed_config()` no longer accepts a latent argument.
- `build_manual_speed_config()` no longer accepts a latent argument.
- `delta_custom` transition resolution requires live H/W supplied by the runtime.
- Sampler selection is handled by the sampler-support layer rather than passing an arbitrary sampler object through the public node path.

If another custom node or script imports `speed_scripts` internals directly, update it against the V2 interfaces before upgrading.

### Scope limits

V2 does **not** add ancestral sampling, SDE variants, CFG++, arbitrary non-H3 models, or evidence-backed parity claims for every supported sampler. Current baked calibration and benchmark evidence are Euler-derived. Re-harvest and benchmark before making quality/parity claims for another sampler or materially different model setup.

### Validation

The V2 branch substantially expands regression coverage around sampler selection, stateful RES lifecycle, host sampler integration, global transition slicing, coincident boundaries, I2V restoration, preview/progress continuity, spectral coupling, Harvest behavior, workflow examples, and all five supported sampler paths.

## 1.x

The original release established the MiniMax-H3 progressive-resolution SPEED implementation with Automatic, Manual, and Euler Sigma Harvest nodes, Euler-only generation, and the initial benchmark/evidence set.
