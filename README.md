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

## Usage

After cloning, load one of these workflows via ComfyUI's workflow browser (Workflow → Open):

- `video_minimax_h3_SPEED.json` — standard SPEED pipeline (sampler → decode → save)
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

Use `video_minimax_h3_SPEED.json` (direct_coarse) for the default path, or `video_minimax_h3_t2v_coupled.json` (coupled_full_grid) when quality matters more than memory.

### Presets (default 20-step schedule)

Each preset splits denoising across resolutions. At full scale the H3 latent is 45×80; each stage runs at `round(45*s) × round(80*s)`. More stages = more time at low res = faster but potentially softer mid-frequency detail.

| Preset | Resolution per stage (45×80 base) | Steps @ each stage | % of 20 | Outcome |
|--------|-----------------------------------|-------------------|---------|---------|
| `half_then_full` | 22×40 → 45×80 | 5 @ 50%, 15 @ 100% | 25% → 75% | Default. Good balance. |
| `three_quarter_then_full` | 34×60 → 45×80 | 10 @ 75%, 10 @ 100% | 50% → 50% | Fastest. Fewer coarse steps, but may miss fine detail. |
| `quarter_half_full` | 11×20 → 22×40 → 45×80 | 3 @ 25%, 2 @ 50%, 15 @ 100% | 15% → 10% → 75% | Higher quality. More refinement passes. |
| `aggressive` | 11×20 → 34×60 → 45×80 | 3 @ 25%, 5 @ 75%, 12 @ 100% | 15% → 25% → 60% | Skips 50% stage. Fast but loses mid-frequency detail. |
| `quarter_half_3q_full` | 11×20 → 22×40 → 34×60 → 45×80 | 3 @ 25%, 2 @ 50%, 3 @ 75%, 12 @ 100% | 15% → 10% → 15% → 60% | Slowest. Highest quality. All intermediate resolutions. |

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
│   ├── oracle.py              — straight-flow oracle for CPU-verified proofs
│   └── tests/                 — 8 test files
├── nodes/
│   ├── sampler_node.py        — MiniMaxH3SPEEDSampler (main)
│   ├── common.py              — shared node plumbing
│   └── helper_nodes/
│       ├── sampler_speed.py   — SPEED as a ComfyUI SAMPLER node
│       ├── sigma_harvest.py   — SigmaHarvest (calibration pass)
│       ├── harvest_to_config.py — parse harvest JSON → report
│       ├── schedule.py        — SpeedConfig planner
│       ├── inspect.py         — debug: inspect latent geometry
│       ├── power_spectrum.py  — debug: radial power spectrum
│       ├── dct_lowpass.py     — debug: DCT lowpass filter
│       ├── transition_math.py — debug: compute transition from A/β
│       ├── spectral_expand.py — debug: visualize spectral expansion
│       ├── x0_fidelity_probe.py — debug: X0 fidelity probe
│       └── av_reentry_oracle.py — debug: AV reentry schedule
└── workflows/
    ├── video_minimax_h3_SPEED.json       — standard SPEED pipeline
    ├── sigma_harvest.json                  — calibration-only workflow
    └── sigma_harvest_calibrated.json       — harvest → report → schedule
```

## Test suite

```bash
uv run pytest minimax_h3_speed/tests/ -q
```

**Current:** 8 test files:
- `test_dct.py` — DCT transform correctness
- `test_debug_nodes.py` — helper node smoke tests
- `test_flow.py` — sigma alignment, audio handling
- `test_harvest.py` — power spectrum, fitting, callback
- `test_integration.py` — end-to-end harvest→schedule→sample pipeline
- `test_oracle.py` — AV reentry oracle
- `test_sampler.py` — sampler node behavior
- `test_spectral.py` — spectral expansion

### Helper Nodes

**Calibration pipeline** (3 nodes):
- `MiniMaxH3SigmaHarvest` (diagnostics) — takes noise, guider, sigmas, latent; returns JSON string
- `MiniMaxH3HarvestToConfig` (diagnostics) — parses harvest JSON into a readable calibration report
- `MiniMaxH3SPEEDSchedule` (schedule) — computes a SpeedConfig from sigmas + preset + mode

**Debug utilities** (7 nodes):
- `MiniMaxH3Inspect` — prints latent shape, device, dtype for troubleshooting
- `MiniMaxH3PowerSpectrum` — computes radial power spectrum of a latent
- `MiniMaxH3DCTLowpass` — applies DCT lowpass filter for ablation studies
- `MiniMaxH3TransitionMath` — computes transition steps from A, β, delta
- `MiniMaxH3SpectralExpand` — visualizes spectral expansion effect on noise
- `MiniMaxH3XFidelityProbe` — measures X0 fidelity during sampling
- `MiniMaxH3AVReentryOracle` — computes when audio should re-enter

Calibration workflow: Run `SigmaHarvest` on a clean noise pass → parse with `HarvestToConfig` → feed calibrated `power_A`/`power_beta` into the sampler in `delta_custom` mode.

PolyForm Noncommercial 1.0.0 — see [LICENSE.md](LICENSE.md).

Canonical SPEED: [howardhx/speed](https://github.com/howardhx/speed).
