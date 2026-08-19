# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0). Free for personal/learning use; contact for commercial use.

A ComfyUI node pack that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSAMPLER + SamplerCustomAdvanced for H3 image-to-video generation.

## Why

Standard diffusion generates at full resolution the whole time. SPEED starts coarse (half or quarter resolution), then progressively refines up to full. You get similar quality with less VRAM and faster generation because most steps run on smaller buffers.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

**Required:** the MiniMax-H3 plugin ([ComfyUI-MiniMax-H3](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3)).

## Nodes

| Node | What it does |
|------|-------------|
| `MiniMaxH3SPEEDSampler` | The sampler. Runs the multi-stage SPEED chain (Euler only) on the H3 latent. |
| `MiniMaxH3HarvestToConfig` | Takes a harvest JSON (STRING input) and emits a readable calibration report. |

The harvest JSON is produced by a **native** (single-resolution) pass, not by the SPEED sampler. Harvested spectra feed the sampler's preset defaults — NOT live wiring. You run the harvest, read the fitted `(A, β)` and sigma recommendations from the report, and bake good values into the node's defaults. SPEED can't harness its own transitions: the per-stage step indexing makes in-sampler spectra meaningless.

## Usage

```
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas       ← BasicScheduler (default 20 steps)
  latent_image ← MiniMaxH3ImageToVideo
  explicit_preset ← "half_then_full" (default)
              ↓
  output → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

Load `workflows/video_minimax_h3_SPEED.json` in ComfyUI's workflow browser.

## Sampler inputs

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `noise` | NOISE | — | |
| `guider` | GUIDER | — | |
| `sigmas` | SIGMAS | — | |
| `latent_image` | LATENT | — | |
| `explicit_preset` | list | `half_then_full` | see Presets |
| `transition_mode` | list | `manual_step` | `manual_step`, `manual_sigma`, `delta_custom` |
| `noise_policy` | list | `direct_coarse` | `direct_coarse`, `coupled_full_grid` |
| `delta` | FLOAT | 0.01 | |
| `power_A` | FLOAT | 150.0 | |
| `power_beta` | FLOAT | 2.0 | |
| `seed_offset` | INT | 10000 | |

Outputs: `(LATENT output, LATENT denoised_output)`.

## Presets

Transition step boundaries for each preset (1-indexed steps out of the 20-step BasicScheduler default):

| Preset | Transition steps |
|--------|-----------------|
| `half_then_full` | 5 |
| `quarter_half_full` | 3, 5 |
| `quarter_half_3q_full` | 3, 5, 8 |
| `aggressive` | 3, 8 |
| `three_quarter_then_full` | 10 |

## Repo layout

```
├── __init__.py               — node registration
├── sampler_node.py           — MiniMaxH3SPEEDSampler
├── harvest_to_config_node.py — MiniMaxH3HarvestToConfig
├── minimax_h3_speed/         — config, flow, runtime, harvest, spectral
├── minimax_h3_speed/tests/   — 4 test files (42 passing)
└── workflows/
    └── video_minimax_h3_SPEED.json
```

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).
