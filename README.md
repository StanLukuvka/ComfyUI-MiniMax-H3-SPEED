# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

⚠️ **Work in progress.** The pack is functional but not yet stable or fully documented.

## What it does

MiniMax-H3 is a video diffusion model that works best when denoising runs at increasing resolution — low-res first, then step up. The SPEED sampler node does exactly this: it runs each stage as its own ComfyUI guider call, DCT-expanding and re-aligning sigma at every boundary. Audio stays unchanged and runs at full resolution throughout.

This replaces KSampler + SamplerCustomAdvanced for H3 workflows, because neither handles mid-flight resolution changes.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

## Nodes

### MiniMaxH3SPEEDSampler

Takes `(noise, guider, sigmas, latent_image)` and runs the multi-stage SPEED chain. Returns `(output_latent, denoised_latent)`.

**Inputs:**

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `noise` | NOISE | — | connect RandomNoise |
| `guider` | GUIDER | — | connect BasicGuider |
| `sigmas` | SIGMAS | — | connect BasicScheduler |
| `latent_image` | LATENT | — | connect the H3 latent |
| `preset` | list | `half_then_full` | one of: `half_then_full`, `quarter_half_full`, `quarter_half_3q_full`, `aggressive`, `three_quarter_then_full` |
| `transition_mode` | list | `manual_step` | `manual_step`, `manual_sigma`, `delta_custom` |
| `noise_policy` | list | `direct_coarse` | `direct_coarse`, `coupled_full_grid` |
| `delta` | FLOAT | 0.01 | Only used in `delta_custom` mode. Range: 1e-4 to 0.5. |
| `noise_amplitude` | FLOAT | 150.0 | Power-spectrum amplitude A. Only used in `delta_custom` mode. |
| `noise_decay_exponent` | FLOAT | 2.0 | Power-law decay beta. Only used in `delta_custom` mode. |
| `seed_offset` | INT | 10000 | Offset added to the noise seed at each resolution boundary. |

`manual_step` and `manual_sigma` both use the preset's hardcoded transition steps. `delta_custom` computes transitions from `(noise_amplitude, noise_decay_exponent)` using a power-law spectral fit.

**Currently:** `sigma_policy` and `audio_policy` are frozen to their canonical defaults in the config — they are not exposed as widget inputs yet.

## Tests

Run with:
```bash
.venv/bin/python -m pytest minimax_h3_speed/tests/ -q
```

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
