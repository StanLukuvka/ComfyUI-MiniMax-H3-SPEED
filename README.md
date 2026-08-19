# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

An experimental ComfyUI node pack that implements progressive-resolution diffusion ([SPEED](https://github.com/howardhx/speed), spectral progressive denoising) for MiniMax-H3 image-to-video latents. The pack's own docstring: it replaces KSampler + SamplerCustomAdvanced because the default sampler does not expect you to change resolution mid-flight — SPEED does exactly that.

## Why

Standard diffusion runs every step at full resolution. SPEED denoises video at low resolution first (cheap steps), then steps the resolution up per preset and continues. Similar quality at lower VRAM and faster generation, because most steps run on smaller buffers. Audio is carried through unchanged at full resolution.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

Dependencies: `networkx`, `numpy`, `torch` (the pack's own imports — ComfyUI provides `torch`). No ComfyUI version requirement.

## Nodes

There are two, registered flat at the repo root (`__init__.py`):

| Node | I/O | Purpose |
|------|-----|---------|
| `MiniMaxH3SPEEDSampler` | in: NOISE, GUIDER, SIGMAS, LATENT + 7 preset/params · out: `(LATENT output, LATENT denoised_output)` | Runs the multi-stage SPEED chain with the Euler sampler (hardcoded — the pack says other samplers need calibration it hasn't done). |
| `MiniMaxH3HarvestToConfig` | in: `harvest_json` (STRING) · out: `(STRING report,)` | Parses a harvest JSON and returns a readable calibration report. |

The harvest JSON is expected to come from a **native** (single-resolution) pass — the node takes it as raw STRING input and it is **not** produced by the SPEED sampler. In-sampler harvesting is circular: SPEED derives its transitions from `(A, β)`, so you cannot fit `(A, β)` from a SPEED run. Run the harvest elsewhere, read the report, and bake good values into the presets' defaults.

## Sampler inputs

All eleven are `required` in `sampler_node.py`:

| Input | Type | Default | Options |
|-------|------|---------|---------|
| `noise` | NOISE | — | connect RandomNoise |
| `guider` | GUIDER | — | connect BasicGuider |
| `sigmas` | SIGMAS | — | connect BasicScheduler |
| `latent_image` | LATENT | — | connect the H3 latent |
| `explicit_preset` | list | `half_then_full` | the five preset keys below |
| `transition_mode` | list | `manual_step` | `manual_step`, `manual_sigma`, `delta_custom` |
| `noise_policy` | list | `direct_coarse` | `direct_coarse`, `coupled_full_grid` |
| `delta` | FLOAT | 0.01 | min 1e-4, max 0.5 |
| `power_A` | FLOAT | 150.0 | min 0, max 1e6 |
| `power_beta` | FLOAT | 2.0 | min 0, max 10 |
| `seed_offset` | INT | 10000 | min 0, max 2³¹−1 |

`manual_step` and `manual_sigma` both resolve to the preset's explicit transition steps; only `delta_custom` uses the `(power_A, power_beta)` thresholds.

## Presets

From `minimax_h3_speed/config.py` — transition step boundaries (1-indexed, of the 20-step BasicScheduler default):

| Preset | Scales | Transition steps |
|--------|--------|-----------------|
| `half_then_full` | 0.5 → 1.0 | 5 |
| `quarter_half_full` | 0.25 → 0.5 → 1.0 | 3, 5 |
| `quarter_half_3q_full` | 0.25 → 0.5 → 0.75 → 1.0 | 3, 5, 8 |
| `aggressive` | 0.25 → 0.75 → 1.0 | 3, 8 |
| `three_quarter_then_full` | 0.75 → 1.0 | 10 |

The sampler raises `ValueError` if the sigma schedule is shorter than the preset requires.

## Tests

```bash
.venv/bin/python -m pytest minimax_h3_speed/tests/ -q
```

4 files, **42 passing** (test_dct, test_flow, test_sampler, test_spectral).

## Repo layout

```
__init__.py               — node registration (sampler + harvest_to_config)
sampler_node.py           — MiniMaxH3SPEEDSampler
harvest_to_config_node.py — MiniMaxH3HarvestToConfig
minimax_h3_speed/         — config, flow, h3_runtime, harvest, spectral
minimax_h3_speed/tests/   — 4 test files
workflows/
  video_minimax_h3_SPEED.json
```

Note: `workflows/video_minimax_h3_SPEED.json` is a **work-in-progress** starter. It currently holds 6 nodes — SaveVideo, three MarkdownNote reference sheets, a ResolutionSelector, and one node whose type is a raw id (`4c314f31…`) that has not been wired to the sampler. Treat it as scratch, not a runnable pipeline.

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
