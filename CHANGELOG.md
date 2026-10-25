# Changelog

All notable changes to this project are documented here. New releases should be added above older ones.

## [2.0.0] - Unreleased

### Added

- Added a `sampler` (SAMPLER) input to Automatic, Manual, and Sigma Harvest, fed by KSamplerSelect or any other SAMPLER node.
- Added a required `model_type` switch (`h3_video`, `flow_image`) to Automatic, Manual, and Sigma Harvest, with no default. `flow_image` runs flow image models such as MOSS Image and Krea 2. Sampler and Harvest both reject non-flow models.
- Added example workflows `workflows/image_moss_SPEED.json` and `workflows/image_krea2_SPEED.json`, and image-model timings (MOSS Image and Krea 2, `coupled_full_grid`, RTX 5070 Laptop) to `evidence/image_evidence/README.md`; split `evidence/` into `h3_evidence/` and `image_evidence/`.
- Added continuous progress and preview reporting across the full SPEED run.
- Replaced the three H3 example workflows (`video_minimax_h3_SPEED_SIGMA.json`, `_Sigma_Calculated.json`, `_Sigma_Manual.json`) with one flat `workflows/video_h3_SPEED.json`, laid out like the image examples: Harvest, Automatic and Manual on one set of loaders, with only Automatic active.
- Added a ComfyUI tooltip to every input and output of Automatic, Manual, and Sigma Harvest.
- Added broader regression coverage for the sampler input, I2V, transitions, Harvest, workflows, and noise policies.

### Changed

- `run_speed_pipeline` now runs family first, then one plan: `families.prepare_family` (new `prepare_family` / `PreparedLatent`) splits the latent, `planning.plan_stages` (new `plan_stages` / `StagePlan`) computes the stage shapes and transition points, and the new `runtime.run_stages` runs the stage loop. No behaviour change (internal: pipeline runs family -> plan -> stages).
- Renamed the `model_type` value `image` to `flow_image`, because only flow (CONST) image models are supported; the class behind it is now `FlowImageFamily` in `families/flow_image.py`. In the same change, `speed_scripts/families.py` became the `speed_scripts/families/` package with one file per family and the registry in `families/__init__.py`.
- Renamed the nodes for the model-agnostic pack: `MiniMaxH3SPEEDSampler` → `SPEEDSampler`, `MiniMaxH3SPEEDSamplerManual` → `SPEEDSamplerManual`, `MiniMaxH3HarvestToConfig` → `SPEEDSigmaHarvest`, display names `SPEED — …`, category `sampling/speed`. The old IDs stay registered as deprecated subclasses and as node replacements, so saved workflows still load and can be swapped in one step. The example workflows use the new IDs.
- `model_type` has no default anywhere: not in the widgets, the node functions, `run_speed_pipeline` or Harvest's `compute_residual`. A prompt without it fails validation instead of running as H3.
- Automatic's `noise_amplitude` and `noise_decay_exponent` no longer default to H3's Euler fit (12.105 / 0.773). They start at 0, which stops with an error asking for a Harvest calibration. The H3 example workflow sets the H3 values explicitly.
- `SpeedConfig` no longer bakes in H3 numbers: `scales` is required, and `delta` / `noise_amplitude` / `noise_decay_exponent` default to `None` and are required only for `delta_custom`.
- Sigma Harvest now defaults to the same 0.005 tolerance as Automatic.
- Sigma Harvest now calibrates the connected sampler instead of assuming Euler.
- Sigma Harvest now reports that its calibration is based on the empirical `x - denoised` residual spectrum.
- Automatic transition planning now uses the live sigma schedule and current latent dimensions.
- Automatic and Manual now share the same planning and validation code.
- I2V keyframes are resized from their original full-resolution copy instead of repeatedly resizing already-scaled latents.
- I2V keyframes are restored before the final stage and after failed runs.
- Harvest now reduces residuals to radial DCT power profiles during sampling instead of keeping every full-resolution residual in memory.
- `coupled_full_grid` now computes the full-grid DCT once per generation and reuses it across transitions.
- The runtime is now separated into dedicated planning and execution modules, with per-model latent handling in `speed_scripts/families.py`. `h3_runtime.py` is now `runtime.py`; `unpack_latent` / `pack_latent` / `resolve_sigma_shifts` moved to `families.py` (`unpack_h3` / `pack_h3`). Harvest's `compute_video_residual` / `extract_video_stream` became `compute_residual` and `families.main_stream`.
- Automatic, Manual, and Harvest now fit a generic empty latent (e.g. `EmptyLatentImage`) to the model's channels before sampling, as SamplerCustomAdvanced does. H3 nested latents are unchanged.
- Harvest calibration JSON now reports `calibration_kind: empirical_residual_fit` and the `model_type`. The Harvest report text now starts with `Empirical <model_type> residual calibration` (for example `Empirical h3_video residual calibration`) instead of `Empirical H3 residual calibration`.

### Fixed

- Sigma Harvest with `model_type h3_video` on a plain 5-D image latent (such as Krea 2's `[B,C,1,H,W]`) no longer calibrates silently; it returns the `no_captures` error that points at `model_type flow_image`.
- `direct_coarse` now draws its transition fill noise on the CPU, as ComfyUI draws its own noise, so a seed gives the same image on every device. Before, it was drawn on the latent's device, so CPU and GPU (and possibly different GPUs) gave different images. This changes `direct_coarse` images for the same seed, often the whole composition, because the fill seeds detail the model builds on; speed is unchanged.
- Fixed Automatic silently coercing invalid stage counts; only 2, 3, or 4 are accepted.
- Fixed non-finite calibration A/beta values passing `SpeedConfig` validation.
- Fixed temporal schedules being able to finish below full temporal resolution.
- Fixed Harvest silently discarding residual/profile reduction exceptions; the first failure is now surfaced.
- Fixed committed Sigma Harvest workflow metadata to match the current two-output node contract.
- Fixed preview callbacks continuing to run after the runtime had logged that preview updates were disabled.
- Fixed unusable Harvest fits being presented as paste-ready Automatic calibration values.
- Fixed repeated I2V resizing accumulating interpolation loss.
- Fixed I2V conditioning remaining modified after some failed runs.
- Fixed progress and previews behaving like separate runs for each SPEED stage.
- Fixed denoised output handling so the MiniMax-H3 video/audio latent structure is preserved.

### Removed

- Removed legacy planner/walker shims, alternate Tolerance aliases, and Harvest compatibility paths.
- Removed V1 Automatic preset names; Automatic now uses only the explicit `stages` selector.
- Removed the old `nodes_common.py` helper layer.
- Removed cached full latent width and height from `SpeedConfig`.
- Removed placeholder Automatic transition indices from runtime configuration.

## [1.x] - Previous releases

- Initial MiniMax-H3 SPEED implementation.
- Automatic and Manual progressive-resolution samplers.
- Euler-only generation and Sigma Harvest.
- Initial benchmark and evidence set.
