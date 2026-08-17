# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md). Free to use for personal/learning projects. Contact for commercial use.

A ComfyUI node that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSAMPLER + SamplerCustomAdvanced.

## Why use this?

Standard diffusion generates at full resolution the whole time. SPEED starts coarse (half or quarter resolution), then progressively refines up to full. You get similar quality with less VRAM and faster generation because most steps run on smaller buffers.

```
MiniMaxH3SPEEDSampler
  noise        ← RandomNoise
  guider       ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas       ← BasicScheduler (default 20 steps)
  latent_image ← MiniMaxH3ImageToVideo
  preset       ← "half_then_full" (default)
                ↓
  output → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

## Install

```bash
git clone https://github.com/StanLukuvka/H3-SPEED.git ComfyUI/custom_nodes/H3-SPEED
# restart ComfyUI
```

**Required:** MiniMax-H3 plugin ([ComfyUI-MiniMax-H3](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3), requires ComfyUI 0.32.0+).

## Usage

After cloning, load one of these workflows via ComfyUI's workflow browser (Workflow → Open):

- `video_minimax_h3_t2v_speed.json` — standard SPEED pipeline (sampler → decode → save)
- `sigma_harvest.json` — calibration pass: harvest residual spectrum from clean noise
- `sigma_harvest_calibrated.json` — full pipeline: harvest → report → schedule node

Options:
- `preset` — see table below
- `transition_mode` — `explicit` (default) or `delta_custom` (uses calibrated A/β)
- `noise_policy` — `direct_coarse` (default) or `coupled_full_grid`

### Noise policies

SPEED reduces VRAM by running early denoising at lower resolution. The `noise_policy` controls how noise is generated across stages:

| Policy | Behavior | When to use |
|--------|----------|-------------|
| `direct_coarse` | Fresh noise per stage, standard SPEED | Default. Lower VRAM, standard quality. |
| `coupled_full_grid` | Shared full-grid noise across all scales | Higher quality (better high-frequency coherence) at the cost of more VRAM. |

Use `video_minimax_h3_t2v_speed.json` (direct_coarse) for the default path, or `video_minimax_h3_t2v_coupled.json` (coupled_full_grid) when quality matters more than memory.

### Presets (default 20-step schedule)

Each preset splits denoising across resolutions. More stages = more time at low res = faster but potentially softer mid-frequency detail.

| Preset | Steps @ each stage | Outcome |
|--------|-------------------|---------|
| `half_then_full` | 5 @ 50%, 15 @ 100% | Default. Good balance. |
| `three_quarter_then_full` | 10 @ 75%, 10 @ 100% | Fastest. Fewer coarse steps, but may miss fine detail. |
| `quarter_half_full` | 3 @ 25%, 2 @ 50%, 15 @ 100% | Higher quality. More refinement passes. |
| `aggressive` | 3 @ 25%, 5 @ 75%, 12 @ 100% | Skips 50% stage. Fast but loses mid-frequency detail. |
| `quarter_half_3q_full` | 3 @ 25%, 2 @ 50%, 3 @ 75%, 12 @ 100% | Slowest. Highest quality. All intermediate resolutions. |

**How to choose:**
- **Speed** → `three_quarter_then_full` (fastest, decent quality)
- **Quality** → `quarter_half_3q_full` (most stages, slowest)
- **Default** → `half_then_full` (proven sweet spot)

## Repository structure

```
H3-SPEED/
├── minimax_h3_speed/
│   ├── config.py              — presets, transition steps, SpeedConfig
│   ├── h3_runtime.py          — multi-stage diffusion loop
│   ├── spectral.py            — resolution expansion math
│   ├── flow.py                — sigma alignment, audio handling
│   ├── harvest.py             — radial power spectrum + fitting
│   └── tests/                 — 61 passing tests
├── nodes/
│   ├── sampler_node.py        — MiniMaxH3SPEEDSampler (main)
│   └── helper_nodes/
│       ├── sigma_harvest.py   — SigmaHarvest (calibration pass)
│       ├── harvest_to_config.py — parse harvest JSON → report
│       └── schedule.py        — SpeedConfig planner
└── workflows/
    ├── video_minimax_h3_t2v_speed.json     — standard SPEED pipeline
    ├── sigma_harvest.json                  — calibration-only workflow
    └── sigma_harvest_calibrated.json       — harvest → report → schedule
```

## Test suite

```bash
PYTHONPATH=minimax_h3_speed python -m pytest minimax_h3_speed/tests/ -q
```

### Helper Nodes

Three companion nodes handle calibration and scheduling:

- `MiniMaxH3SigmaHarvest` (diagnostics) — takes noise, guider, sigmas, latent; returns JSON string
- `MiniMaxH3HarvestToConfig` (diagnostics) — parses harvest JSON into a readable calibration report
- `MiniMaxH3SPEEDSchedule` (schedule) — computes a SpeedConfig from sigmas + preset + mode

Calibration workflow: Run `SigmaHarvest` on a clean noise pass → parse with `HarvestToConfig` → feed calibrated `power_A`/`power_beta` into the sampler in `delta_custom` mode.

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).

Canonical SPEED: [howardhx/speed](https://github.com/howardhx/speed).
