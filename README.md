# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial** — [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0)

> *"Why make big noise when little noise do trick?"*

Make MiniMax-H3 video faster without re-training. Starts the denoise at low resolution (cheap), then upsamples to full resolution when detail actually matters. Same prompt, same seed — just fewer full-res steps.

> **Only Euler, only MiniMax-H3.** Audio is always full-resolution.

## Quickstart — 3 steps, no tuning

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

1. Load `workflows/video_minimax_h3_SPEED_Sigma_Calculated.json` (or any H3 text-to-video workflow).
2. Replace your `KSampler` / `SamplerCustomAdvanced` with **MiniMax H3 SPEED — Sampler (Automatic)**. Wire the same `noise`, `guider`, `sigmas`, `latent_image`.
3. Set **`stages = 2`** (fastest) or **`3`** (balanced, default) and hit Queue. It works out of the box — no Harvest needed.

That's it. The baked calibration does the rest.

## Which node do I need?

**Automatic — Sampler (99% of users)**
Just `stages` (2, 3 or 4). `2 = 0.5→1.0` fastest, `3 = 0.33→0.66→1.0` default, `4 = 0.25→0.5→0.75→1.0` most conservative. Leave `Tolerance (Delta)`, `noise_amplitude`, `noise_decay_exponent` at defaults unless you change checkpoint.

*Hover text:* "Automatic SPEED sampler — just pick how many stages (2, 3 or 4) and go. Starts cheap at low-res and upsamples when detail matters."

**Manual — Sampler (Step-Through)**
You set up to four `(goal, resolution)` pairs yourself. `goal` = step where that stage ends, `resolution` = scale like `0.25` = quarter. Set `goal` or `resolution` to `0` to skip a stage. Use only to copy a paper schedule or test a custom ladder.

*Hover text:* "Manual SPEED sampler — set stages by hand. Goal 0 or resolution 0 skips a stage."

**Sigma Harvest (Native Euler)**
Run **once** at full resolution to measure your checkpoint if you change model. It gives you `A / beta` to paste into Automatic. 
If you are using Loras, or other models/addons/optimisations that influence the function of the model I recommend running it to see to ensure it is optimized for your specific workload.

You can instead use the following values for base H3:

- **Default (baked, 1%):** `Tolerance (Delta)=0.01, noise_amplitude=7.394, noise_decay_exponent=0.62` — `r²0.60 fair`, balanced for `stages 2`
- **Conservative (0.5%):** `Tolerance (Delta)=0.005, noise_amplitude=12.454, noise_decay_exponent=0.819` — `r²0.70 good`, sharper for `stages 3` if text wobbles

*Hover text:* "Sigma Harvest — run once at full-res to calibrate Automatic. Measures noise falloff."

Workflow wires are the same for all three: `noise` → `guider` → `sigmas` → `latent_image` → `output_latent` → `VAE Decode`.

## How fast is it? What breaks?

10s 0.5MP Office "world's most mediocre boss" mug clip, same seed:

**Default fit (`Δ0.01 A7.394 β0.62`):**
| Mode | Time | Quality |
|------|------|---------|
| Native (no SPEED) | 833s cold | reference |
| 2-stage `direct` | 651s cold | good — prompt intact |
| 2-stage `coupled` | 608s | good, slightly sharper text |
| 3-stage `direct` | 415s | **blurry** — needs `coupled` |
| 3-stage `coupled` | 616s | sharp again, but no faster than 2-stage |
| 4-stage `direct` | 262s | **unusable** |
| 4-stage `coupled` | 608s | sharp but prompt drifts |

**Conservative fit (`Δ0.005 A12.454 β0.819`, optional):**
| Mode | Time | Quality |
|------|------|---------|
| 2-stage `direct` | 672s | good — prompt intact |
| 3-stage `direct` | 540s | good — quality holds without `coupled` |
| 4-stage `direct` | 400s | usable, slight halo |

`direct_coarse` = fastest, lowest VRAM. `coupled_full_grid` = ~30-50% slower, can rescue 3-stage text. See [evidence/README.md](evidence/README.md) for full 10s GIFs (360p 12fps) and mp4s.

**Rule of thumb:** Use `stages 2`. Try `3` only if you need that last bit of text/sharpness and can pay `coupled`.

## Troubleshooting

- **"Sigma schedule too short"** → increase `BasicScheduler` steps. Need at least `stages × 2` sigmas (e.g. stages 3 needs ≥6 steps).
- **"H3 model required"** → this only works with a real MiniMax-H3 model (has `sigma_shift_video` / `sigma_shift_audio`). Not SD/Flux/WAN.
- **Text looks blurry / wobbly** → try `noise_policy = coupled_full_grid`, or lower `Tolerance (Delta)` from `0.01` (1%) to `0.005` (0.5% — more conservative, slower but sharper).
- **Prompt drifts / objects disappear on 4-stage** → too many hops. Drop to 2 or 3 stages.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

It measures how noise power falls with frequency on a full-res run: `P(ω) = A·|ω|^-β` (β ~0.6 for MiniMax-H3). For each scale `s`, `ω = s·min(H,W)/2`, `P = A·ω^-β`, then `thr = 1/(1+√(δ/(P·(1+P-δ))))` (δ = Tolerance, 0.01 = 1% allowed error). The first `sigmas[i] ≤ thr` is where that stage ends. Continuous sigma, just quantized to your sigma schedule.

Baked default `A7.394 β0.62 δ0.01` came from a 0.6MP 40-step `simple` harvest (`r²0.60 fair`, stable across 0.5→0.6MP). More conservative fit `A12.45 β0.819 δ0.005 r²0.70 good` available if text wobbles. Re-calibrate with the Harvest node if you change checkpoint: wire `noise/guider/sigmas/latent + Tolerance`, run a native Euler generation at 28-32 steps `simple`, copy `calibration` JSON into Automatic's `noise_amplitude` / `noise_decay_exponent` / `Tolerance`.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`. Old `preset` names (`half_then_full` etc) still load and map to stages.

`seed_offset` decorrelates the per-stage high-frequency fill from your starting noise — leave at 10000 unless you want a different fill pattern for the same seed. `ratio_mode steps` = goal is a step index, `ratio` = goal is 0-1 fraction.

</details>

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
