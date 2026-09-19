# Changelog

All notable changes to this project are documented here. New releases should be added above older ones.

## [2.0.0] - Unreleased

### Added

- Added sampler selection to Automatic, Manual, and Sigma Harvest.
- Added support for Euler, Heun, DPM2, Exp Heun 2 X0, and RES Multistep.
- Added a run-scoped RES Multistep adapter.
- Added continuous progress and preview reporting across the full SPEED run.
- Added broader regression coverage for sampler selection, RES state, I2V, transitions, Harvest, workflows, and noise policies.

### Changed

- Sigma Harvest now defaults to the same 0.005 tolerance as Automatic.
- Sigma Harvest now calibrates the selected native sampler instead of assuming Euler.
- Sigma Harvest now reports that its calibration is based on the empirical `x - denoised` residual spectrum.
- Automatic transition planning now uses the live sigma schedule and current latent dimensions.
- Automatic and Manual now share the same planning and validation code.
- I2V keyframes are resized from their original full-resolution copy instead of repeatedly resizing already-scaled latents.
- I2V keyframes are restored before the final stage and after failed runs.
- Harvest now reduces residuals to radial DCT power profiles during sampling instead of keeping every full-resolution residual in memory.
- `coupled_full_grid` now computes the full-grid DCT once per generation and reuses it across transitions.
- The runtime and sampler-specific behavior are now separated into dedicated planning, sampler-support, and RES adapter modules.

### Fixed

- Fixed committed Sigma Harvest workflow metadata to match the current two-output node contract.
- Fixed preview callbacks continuing to run after the runtime had logged that preview updates were disabled.
- Fixed unusable Harvest fits being presented as paste-ready Automatic calibration values.
- Fixed RES history being unsafe to carry across resolution changes by resetting it at every SPEED transition.
- Fixed repeated I2V resizing accumulating interpolation loss.
- Fixed I2V conditioning remaining modified after some failed runs.
- Fixed progress and previews behaving like separate runs for each SPEED stage.
- Fixed denoised output handling so the MiniMax-H3 video/audio latent structure is preserved.

### Removed

- Removed the legacy `speed_scripts/automatic_config.py` compatibility module, guider walker marker, alternate Tolerance aliases, and Harvest compatibility wrapper.
- Removed V1 Automatic preset names; Automatic now uses only the explicit `stages` selector.
- Removed the old `nodes_common.py` helper layer.
- Removed cached full latent width and height from `SpeedConfig`.
- Removed placeholder Automatic transition indices from runtime configuration.

### Compatibility

- Public node IDs remain unchanged.
- Automatic and Manual still default to Euler when no sampler is specified.
- Existing Automatic and Manual input ordering is preserved, with `sampler_name` appended.
- V1 Automatic preset names and alternate Tolerance argument aliases were removed in V2.
- Internal `speed_scripts` APIs changed and should not be treated as V1-compatible.

## [1.x] - Previous releases

- Initial MiniMax-H3 SPEED implementation.
- Automatic and Manual progressive-resolution samplers.
- Euler-only generation and Sigma Harvest.
- Initial benchmark and evidence set.
