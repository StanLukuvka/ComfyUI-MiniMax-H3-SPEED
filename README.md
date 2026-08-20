# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

⚠️ **Work in progress.** The pack is functional but not yet stable or fully documented.

⚠️ **Text-to-video only.** Image-to-video is not supported yet.

## What it does

SPEED (Spectral Progressive Diffusion for Efficient image and video generation) is a technique from the [SPEED paper](https://github.com/howardhx/speed). The idea: instead of running all 20+ denoising steps at full 720p resolution, start at a fraction (say 25%) and only step up to full resolution at preset-determined boundaries. Early denoising steps don't need full resolution — low-frequency structure emerges first — so the coarse stages produce the same result at a fraction of the compute and VRAM.

This node implements that for MiniMax-H3's nested video+audio latents: each resolution stage is a separate `guider.sample()` call, with a DCT-based spectral expansion to upsample between stages and kappa sigma-alignment at each boundary. The audio track is carried through at full resolution unchanged.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

## How to use

1. Install the pack (see below).
2. In ComfyUI, load the example workflow: `workflows/video_minimax_h3_SPEED.json`.
3. Wire your H3 model into the guider and hit run.

The workflow is a starter — it has the sampler node and the standard ComfyUI nodes (RandomNoise, BasicGuider, BasicScheduler) already laid out. Adjust the preset and scheduler steps to taste.

## Inputs

| Input | Type | Default | Notes |
|-------|------|---------|-------|
| `noise` | NOISE | — | Standard ComfyUI noise node |
| `guider` | GUIDER | — | Your H3 model's guider |
| `sigmas` | SIGMAS | — | Standard ComfyUI scheduler |
| `latent_image` | LATENT | — | H3 video latent |
| `preset` | list | `half_then_full` | See table above |
| `transition_mode` | list | `manual_step` | `manual_step`, `manual_sigma`, `delta_custom` |
| `noise_policy` | list | `direct_coarse` | Usually leave as default |
| `delta` | FLOAT | 0.01 | Only matters in `delta_custom` mode |
| `noise_amplitude` | FLOAT | 150.0 | Only matters in `delta_custom` mode |
| `noise_decay_exponent` | FLOAT | 2.0 | Only matters in `delta_custom` mode |
| `seed_offset` | INT | 10000 | Added to seed at each resolution boundary |

**Output:** `(output_latent, denoised_latent)` — connect `output_latent` to your save node. `denoised_latent` is the clean final result if you need it for further processing.

## Troubleshooting

- **Sigma schedule too short:** If you get a ValueError about sigma schedule length, increase your `BasicScheduler` steps. Each preset needs at least `n_stages * 2` sigmas.
- **H3 model required:** This sampler requires a real MiniMax-H3 model with `sigma_shift_video` / `sigma_shift_audio` attributes. It won't work with SD, Flux, WAN, or other model types.

## TODO

<!-- Evidence and benchmark times — VRAM usage, generation time, quality
comparisons vs standard KSampler, per-preset breakdowns. To be filled in
once real measurements are collected on actual H3 hardware. -->

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
