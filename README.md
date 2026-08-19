# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md). Free for personal/learning projects; contact for commercial use.

A ComfyUI node that runs [SPEED](https://github.com/howardhx/speed) (Spectral Progressive Diffusion) on MiniMax-H3's packed video+audio latent. Replaces KSampler + SamplerCustomAdvanced for MiniMax-H3 video generation.

## Why use this?

Standard diffusion generates at full resolution the whole time. SPEED starts coarse (half or quarter resolution), then progressively refines up to full resolution. You get similar quality with less VRAM and faster generation because most steps run on smaller buffers.

## Install

ComfyUI Manager (recommended), or manually:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
cd ComfyUI-MiniMax-H3-SPEED
uv run pytest minimax_h3_speed/tests/ -q   # optional sanity check
# restart ComfyUI
```

**Required:** MiniMax-H3 plugin ([ComfyUI-MiniMax-H3](https://github.com/StanLukuvka/ComfyUI-MiniMax-H3), requires ComfyUI 0.32.0+).

## Nodes

This pack ships **two nodes**:

| Node | Display name | Inputs | Outputs |
|------|-------------|--------|---------|
| `MiniMaxH3SPEEDSampler` | MiniMax H3 SPEED — Sampler | `noise` (NOISE), `guider` (GUIDER), `sigmas` (SIGMAS), `latent_image` (LATENT), `explicit_preset`, `transition_mode`, `noise_policy`, `delta`, `power_A`, `power_beta`, `seed_offset` | `output` (LATENT), `denoised_output` (LATENT) |
| `MiniMaxH3HarvestToConfig` | MiniMax H3 SPEED — Harvest → Config | `harvest_json` (STRING) | `report` (STRING) |

```text
MiniMaxH3SPEEDSampler
  noise         ← RandomNoise
  guider        ← BasicGuider (UNETLoader + MiniMaxH3ImageToVideo)
  sigmas        ← BasicScheduler (default 20 steps)
  latent_image  ← MiniMaxH3ImageToVideo
  explicit_preset ← "half_then_full" (default)
              ↓
  output → VAEDecode + VAEDecodeAudio → CreateVideo → SaveVideo
```

### Presets

Each preset splits denoising across resolutions. More stages = more time at low res = faster but potentially softer mid-frequency detail.

| Preset | Scales | Transition steps | Outcome |
|--------|--------|-----------------|---------|
| `half_then_full` | 50% → 100% | step 5 | Default. Good balance. |
| `three_quarter_then_full` | 75% → 100% | step 10 | Fastest. Fewer coarse steps. |
| `quarter_half_full` | 25% → 50% → 100% | steps 3, 5 | More refinement passes. |
| `aggressive` | 25% → 75% → 100% | steps 3, 8 | Skips 50%. Fast, loses mid detail. |
| `quarter_half_3q_full` | 25% → 50% → 75% → 100% | steps 3, 5, 8 | Slowest. Highest quality. |

**How to choose:**
- **Speed** → `three_quarter_then_full`
- **Quality** → `quarter_half_3q_full`
- **Default** → `half_then_full`

### Transition mode

- `manual_step` / `manual_sigma` — explicit preset transition steps (default behavior; same thing here)
- `delta_custom` — uses `power_A` / `power_beta` (calibrated from a native-sampler harvest) for δ-optimal transitions

### Noise policy

| Policy | Behavior |
|--------|----------|
| `direct_coarse` | Fresh noise per stage, standard SPEED. Lower VRAM. |
| `coupled_full_grid` | Shared full-grid noise across scales. Higher quality, more VRAM. |

## Sigma harvesting (calibration)

The pack does **not** harvest inside the SPEED sampler. To get `(A, β)` for `delta_custom`, run a **native** single-resolution Euler pass with a harvest callback, then feed the JSON into `MiniMaxH3HarvestToConfig` to read the fitted values. See [AGENTS.md](AGENTS.md) for why this must not run inside SPEED (multi-stage sigma splicing mislabels per-step sigmas).

## Workflows

Load via ComfyUI's workflow browser (Workflow → Open):

- `workflows/video_minimax_h3_SPEED.json` — standard SPEED pipeline (sampler → decode → save)
- `workflows/video_minimax_h3_t2v_speed.json` — `direct_coarse` noise policy
- `workflows/video_minimax_h3_t2v_coupled.json` — `coupled_full_grid` noise policy

## Repository structure

```text
ComfyUI-MiniMax-H3-SPEED/
├── __init__.py                  — registration (2 nodes)
├── sampler_node.py              — MiniMaxH3SPEEDSampler
├── harvest_to_config_node.py    — MiniMaxH3HarvestToConfig
├── minimax_h3_speed/
│   ├── config.py                — presets, transition steps, SpeedConfig
│   ├── h3_runtime.py            — multi-stage diffusion loop
│   ├── spectral.py              — resolution expansion math
│   ├── flow.py                  — sigma alignment, audio handling
│   ├── harvest.py               — power spectrum + power-law fitting
│   └── tests/                   — 4 test files, 42 passing
├── workflows/                   — 3 ready-to-load workflows
├── AGENTS.md                    — correctness constraints
└── NEXT.md                      — roadmap
```

## License

PolyForm Noncommercial 1.0.0 — free for personal/learning, contact for commercial use. See [LICENSE.md](LICENSE.md).
