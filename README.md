# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

⚠️ **Work in progress.** The pack is functional but not yet stable or fully documented.

⚠️ **Text-to-video only.** Image-to-video is not supported yet.

## What it does

SPEED (Spectral Progressive Diffusion for Efficient image and video generation) is a technique from the [SPEED paper](https://github.com/howardhx/speed). 

The idea: instead of running all 20+ denoising steps at full 720p resolution, start at a fraction (say 25%) and only step up to full resolution at preset-determined boundaries. 

Early denoising steps don't gain anything from full resolution as only low-frequency structure emerges first. So a lower resolution stage produce the same result at a fraction of the compute and VRAM.

This node implements that for MiniMax-H3's nested video+audio latents: each resolution stage is a separate `guider.sample()` call, with a DCT-based spectral expansion to upsample between stages and kappa sigma-alignment at each boundary. Only Euler sampler is supported — other samplers need calibration that hasn't been done.

The audio track is carried through at full resolution unchanged.

## Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

## How to use

1. Install the pack (see above).
2. In ComfyUI, load the example workflow: `workflows/video_minimax_h3_SPEED.json`.
3. Wire your H3 model into the guider and hit run.

The workflow is a starter — it has the sampler node and the standard ComfyUI nodes (RandomNoise, BasicGuider, BasicScheduler) already laid out. Adjust the preset and scheduler steps to taste.

**Presets** (the `preset` widget):

- `half_then_full` — start at 50%, finish full (recommended default)
- `quarter_half_full` — start at 25%, step to 50%, finish full
- `quarter_half_3q_full` — start at 25%, step to 50%, then 75%, finish full
- `aggressive` — start at 25%, jump to 75%, finish full
- `three_quarter_then_full` — start at 75%, finish full

**Transition mode** (the `transition_mode` widget):

- `manual_step` — use the preset's hardcoded step boundaries (default)
- `manual_sigma` — same boundaries, resolved by sigma value instead of step index
- `delta_custom` — compute boundaries at runtime from a power-law spectral fit using `noise_amplitude`, `noise_decay_exponent`, and `delta`. Only for advanced tuning.

Everything else on the node is standard ComfyUI wiring (noise, guider, sigmas, latent). The output is `(output_latent, denoised_latent)` — connect `output_latent` to your save node.

## Troubleshooting

- **Sigma schedule too short:** If you get a ValueError about sigma schedule length, increase your `BasicScheduler` steps. Each preset needs at least `n_stages * 2` sigmas.
- **H3 model required:** This sampler requires a real MiniMax-H3 model with `sigma_shift_video` / `sigma_shift_audio` attributes. It won't work with SD, Flux, WAN, or other model types.

## Video and Timings

TODO

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
