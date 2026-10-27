# AGENTS.md — ComfyUI-MiniMax-H3-SPEED-Sampler

Rules for working on this pack. Usage, reference A/beta values and per-sampler results: `README.md`. Migration notes: `CHANGELOG.md`.

## Public contract (do not break)

- Three nodes: `SPEEDSampler` (Automatic), `SPEEDSamplerManual` and `SPEEDSigmaHarvest`. The old IDs `MiniMaxH3SPEEDSampler`, `MiniMaxH3SPEEDSamplerManual` and `MiniMaxH3HarvestToConfig` stay registered as `DEPRECATED` subclasses with node replacements, so saved workflows still load. Keep both sets stable.
- All three nodes take a standard ComfyUI `SAMPLER` input. Any sampler is accepted: no allowlist, no per-sampler adapter. Each stage is its own `guider.sample()` call.
- All three take a required `model_type` (`h3_video`, `flow_image`) with no default anywhere: widget, node signature, runtime or Harvest helpers. A missing value fails validation and a mismatched one raises.
- Nothing model-specific is a default. Automatic's `noise_amplitude` and `noise_decay_exponent` start at 0 (unset, raises), and `SpeedConfig` has no baked A/beta/delta.
- `direct_coarse` is the default noise policy. Keep `coupled_full_grid` available.
- Workflows use native ComfyUI widget slugs (`NOISE`, `GUIDER`, `SAMPLER`, `SIGMAS`, `LATENT`).

## Where things live

- `speed_scripts/families/`: one class per `model_type`. `__init__.py` has the registry (`MODEL_TYPES`, `get_family`, `check_model_type`, `prepare_latent`, `prepare_family`). `h3_video.py` has `H3VideoFamily` with its unpack/pack, sigma-shift and audio-carry helpers. `flow_image.py` has `FlowImageFamily` with `require_flow_sampling`. A family splits the latent into the spatial stream SPEED resizes (always `[B,C,T,H,W]` in the runtime) and an optional extra stream, and implements `to_internal`, `carry_extra` and `main_stream`.
- `speed_scripts/planning.py`: stage geometry, transition-threshold math, `plan_stages`, Automatic config construction and Manual schedule normalization.
- `speed_scripts/runtime.py`: execution only. `run_speed_pipeline` runs `families.prepare_family` -> `planning.plan_stages` -> `run_stages`. Coupled full-grid noise is transformed to spectral coefficients once per generation.
- `speed_scripts/latent_class.py`: `LatentWalker`, local to one `run_speed_pipeline()` call and never stored on the guider. It snapshots only keyframe latents, always resizes from the pristine full-resolution copy, restores them before the final stage and on failure, and never wraps or resizes `minimax_refs`.
- Harvest (`nodes/sampler_sigma_harvest_node.py`, `speed_scripts/harvest.py`) runs the connected native sampler, not the SPEED pipeline. It picks its family with `get_family` like the samplers. It reduces each residual to a CPU radial power profile inside the callback and keeps no full residual tensors.
- Automatic configs store no placeholder transition indices or latent sizes. `delta_custom` boundaries come from the live sigmas and latent at runtime. `transition_steps` is only for explicit/Manual mode.

## Rules

- Add a model by adding a family, not by branching in the runtime. `model_type` is never auto-detected.
- No random configuration and no silent randomization in config paths.
- Euler is the calibration and benchmark reference. Do not claim parity or recommend a sampler without a measured run. Known-bad samplers go in the README.

## Tests

`.venv/bin/python -m pytest -q speed_scripts/tests`. There is no CI on dev PRs, so this local suite is the gate.
