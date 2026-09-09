# ComfyUI MiniMax-H3 SPEED Sampler

⚠️ **Noncommercial** — [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0) 
(I don't expect this to be used commercially. If it genuinely will be, message me.)



> *"Why make big noise when little noise do trick?"*

Make MiniMax-H3 video faster without re-training. 
Starts the denoise at low resolution, then upsamples to full resolution when finetuned detail starts appearing within noise. Allowing us to save on generations.

> **Only Euler, only MiniMax-H3.** Audio is always full-resolution.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

1. Replace your `KSampler` / `SamplerCustomAdvanced` with **MiniMax H3 SPEED — Sampler (Automatic)**. Wire the same `noise`, `guider`, `sigmas`, `latent_image`.
2. Set **`stages = 2`** (fastest) or **`3`** (balanced, default) and hit Queue. Current default settings are the conservative sigma harvest at 0.5% delta.


## Which node do I need?

**Automatic — Sampler**
Just `stages` (2, 3, or 4) that correspond to how many resolution stages there are. 
`2 = 0.5→1.0`, 
`3 = 0.33→0.66→1.0`, 
`4 = 0.25→0.5→0.75→1.0`. 

`Tolerance (Delta)`, `noise_amplitude`, `noise_decay_exponent` determine at what steps each stage is triggered at, generally leave unless experimenting.

**Manual — Sampler (Step-Through)**
You set up to four `(goal, resolution)` pairs yourself. `goal` = step where that stage ends, `resolution` = scale like `0.25` = quarter. Set `goal` or `resolution` to `0` to skip a stage. Use only to copy a paper schedule or test a custom ladder.


**Sigma Harvest (Native Euler)**
Run **once** with your current workflow to measure your checkpoint. It gives you `A / β` to paste into Automatic.
If you are using LoRAs, or other models, addons, or optimisations that change how the model behaves, I recommend running it to ensure it is tuned to your specific workload.

You can instead use the following values for base H3:

- **Default (baked, 0.5%):** `Tolerance (Delta)=0.005, noise_amplitude=12.105, noise_decay_exponent=0.773` — `r² 0.70`
- **Balanced (1%):** `Tolerance (Delta)=0.01, noise_amplitude=12.436, noise_decay_exponent=0.786` — near parity, faster

See the [evidence section](evidence/README.md) for what changes in generation.

Workflow wires are the same for all three: `noise` → `guider` → `sigmas` → `latent_image` → `output_latent` → `VAE Decode`.

## Diagnostics

- **Sigma Harvest (Native Euler)** runs one native full-res Euler pass and outputs one aggregate residual calibration (`A / β`) to paste into Automatic.

## Speed Improvements

Same 10s 0.5MP "world's most mediocre boss" office mug clip, same seed, corrected scheduler (post-PR-#37). Native Euler baseline: 571s. Per-resolution harvest calibrations were used for each fit.

| Fit | Mode | Time | Speedup | Quality |
|------|------|------|---------|---------|
| Δ0.005 `A12.105 β0.773` | 2-stage | 463s | 1.23× | native equivalent|
| Δ0.005 | 3-stage | 439s | 1.30× | native equivalent |
| Δ0.005 | 4-stage | 435s | 1.31× | native equivalent, mildest melt artifact |
| Δ0.01 `A12.436 β0.786` | 2-stage | 450s | 1.27× | near parity |
| Δ0.01 | 3-stage | 410s | 1.39× | cleanest mug-landing beat |
| Δ0.01 | 4-stage | 384s | 1.49× | inconsistencies start appearing |
| Δ0.05 `A6.920 β0.766` | 2-stage | 278s | 2.05× | noticable artifacting |
| Δ0.05 | 3-stage | 262s | 2.18× | very noticable artifacting but still usable |
| Δ0.05 | 4-stage | 238s | 2.41× | intense artifacting and halo effect beginning |


See [evidence/README.md](evidence/README.md) for full 10s GIFs (360p 12fps) and the review rubric.

**Rule of thumb:** quality-first use `stages 3` at Δ0.005; balanced use `stages 2` at Δ0.01; fast drafts use `stages 4` at Δ0.05.

## Troubleshooting

- **"Sigma schedule too short"** → increase `BasicScheduler` steps. The last stage boundary must leave at least one denoising step: with the final boundary at step `g`, you need ≥ `g + 2` sigmas (e.g. a 4-stage run with boundaries 3/5/8 needs ≥10 sigmas = 9 steps).
- **"H3 model required"** → this only works with a real MiniMax-H3 model (one that has `sigma_shift_video` / `sigma_shift_audio`). Not SD/Flux/WAN.
- **Text looks blurry / wobbly** → try `noise_policy = coupled_full_grid`, or lower `Tolerance (Delta)` from `0.005` (0.5%) to `0.001` — more conservative, slower but sharper.
- **Prompt drifts / objects disappear on 4-stage** → too many hops. Drop to 2 or 3 stages.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

It measures how noise power falls with frequency on a full-res run: `P(ω) = A·|ω|^-β` (β ~0.77 for MiniMax-H3's validated fits; see Defaults below). For each scale `s`, `ω = s·min(H,W)/2`, `P = A·ω^-β`, then `thr = 1/(1+√(δ/(P·(1+P-δ))))` (δ = Tolerance, 0.005 = 0.5% allowed error). The first `sigmas[i] ≤ thr` is where that stage ends. Continuous sigma, just quantized to your sigma schedule.

Re-calibrate with the Harvest node if you change checkpoint: wire `noise/guider/sigmas/latent + Tolerance`, run a native Euler generation at 28-32 steps with `sampler = simple`, copy `calibration` JSON into Automatic's `noise_amplitude` / `noise_decay_exponent` / `Tolerance`.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`.

`seed_offset` changes the per-stage high-frequency fill pattern — leave at 10000 unless you want a different fill pattern for the same seed. `ratio_mode steps` = goal is a step index, `ratio` = goal is a 0-1 fraction.

</details>

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
