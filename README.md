# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial license** — see [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0).

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

Three nodes:

**1. MiniMax H3 SPEED — Sampler (Automatic)** — the default. `stages` picks how many resolution stages (2-4, evenly spaced), `Tolerance (Delta)` + `noise_amplitude`/`noise_decay_exponent` auto-compute step boundaries from the power spectrum (`P=A·ω^-β`, `thr=1/(1+√(δ/(P(1+P-δ))))`). Baked calibration is `A7.394 β0.62 δ0.01` from a 0.6MP 40-step harvest (β0.59-0.62 stable across 0.5→0.6MP, 20/40 steps). Just set stages=3 and go.

**2. MiniMax H3 SPEED — Sampler (Manual Step-Through)** — explicit control. No `Tolerance`/`A`/`β` (those are for auto). Set `ratio_mode steps` (step indices) or `ratio` (fraction of schedule) and up to four `(goal, resolution)` pairs — `goal 0` disables that stage. Needs at least two active stages ending at `1.0`.

**3. MiniMax H3 SPEED — Sigma Harvest (Native Euler)** — calibrates `A/β` for the automatic sampler. Wire `noise`/`guider`/`sigmas`/`latent_image` + `Tolerance (Delta)`, run a native full-res generation, copy `calibration` JSON (`noise_amplitude`, `noise_decay_exponent`, `delta`, `r²`, `health`) into the Automatic sampler. Report includes plug-and-play line and diagnostic `Reference 0.50x/0.75x → sigma~X [thr Y]`. Use `simple` scheduler, 28-32 steps for a clean `r²>0.6`.

Load the example workflow: `workflows/video_minimax_h3_SPEED.json`.

**Stages** (the `stages` widget on Automatic):
- `2` — 0.5 → 1.0
- `3` — 0.33 → 0.66 → 1.0 (default)
- `4` — 0.25 → 0.5 → 0.75 → 1.0

2 is fastest, 4 is slowest/most conservative. Steps within each stage are auto-placed from `Tolerance (Delta)` + `A/β` via `thr=1/(1+√(δ/(P(1+P-δ))))` (continuous sigma, quantized to your `sigmas`).

*Old workflows with `preset` (half_then_full etc) still load — mapped to stages (2→2, quarter_half_full/aggressive→3, quarter_half_3q_full→4).*

**Manual scales** — set via `ratio_mode` + 4× `(goal, resolution)` pairs on Manual (explicit), not preset.

**Noise policy** (`noise_policy` on both samplers):
- `direct_coarse` (default) — fresh DCT noise per stage, lowest VRAM
- `coupled_full_grid` — one full-res noise grid reused across stages (DCT-coupled), slightly better temporal consistency at upscale, higher VRAM

Everything else is standard ComfyUI wiring (noise, guider, sigmas, latent). Output is `(output_latent, denoised_latent)` — connect `output_latent` to save/decode.

## Troubleshooting

- **Sigma schedule too short:** If you get a ValueError about sigma schedule length, increase your `BasicScheduler` steps. Each stages setting needs at least `n_stages * 2` sigmas.
- **H3 model required:** This sampler requires a real MiniMax-H3 model with `sigma_shift_video` / `sigma_shift_audio` attributes. It won't work with SD, Flux, WAN, or other model types.

## Video and Timings

10s 0.5MP Office mug clip (same seed, `Δ0.01 A7.394 β0.62`, baked default):

| Mode | Time | Quality | Notes |
|------|------|---------|-------|
| Native | 833s (cold) | reference | full-res Euler |
| 2-stage `direct` | 651s cold | good | prompt intact |
| 2-stage `coupled` | 608s | good, slightly sharper | `Mediocre` correct vs `Medlooae` on direct |
| 3-stage `direct` | 415s | **blurry** | notably destroys, fastest but unusable |
| 3-stage `coupled` | 616s | **restored** | quality back, but slower than direct |
| 4-stage `coupled` | 608s | sharp but prompt drifts | too many hops, not worth it |

`direct_coarse` = fastest, lowest VRAM. `coupled_full_grid` = ~30-50% slower, holds high-ω text on 3-stage. See [`evidence/README.md`](evidence/README.md) for side-by-side GIFs and mp4s. Pending: `Δ0.005 A12.45 β0.819 r²0.70 good` (more conservative, may hold on `direct`).

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
