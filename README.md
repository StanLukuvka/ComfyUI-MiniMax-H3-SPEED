# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

⚠️ **Work in progress.** The pack is functional but not yet stable or fully documented.

## What it does

MiniMax-H3 is a video diffusion model. Instead of running every denoising step at full resolution (which wastes VRAM and time), this node runs the cheap low-res steps first, then steps resolution up at the right moments. Same quality, less memory, faster.

Audio always runs at full resolution — nothing to configure there.

Replaces KSampler + SamplerCustomAdvanced because those don't handle mid-flight resolution changes.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

## How to use

Drop `MiniMax H3 SPEED — Sampler` into your ComfyUI graph and wire it like you would a normal sampler:

1. **NOISE** — connect `RandomNoise`
2. **GUIDER** — connect `BasicGuider` (with your H3 model)
3. **SIGMAS** — connect `BasicScheduler`
4. **LATENT_IMAGE** — connect your H3 video latent
5. **PRESET** — pick one:
   - `half_then_full` — start at 50%, finish full (recommended default)
   - `quarter_half_full` — start at 25%, step to 50%, finish full
   - `aggressive` — start at 25%, jump to 75%, finish full
   - `three_quarter_then_full` — start at 75%, finish full
6. **TRANSITION_MODE** — `manual_step` (use preset defaults) or `delta_custom` (compute from spectral analysis)
7. Leave everything else as default unless you're tuning.

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

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
