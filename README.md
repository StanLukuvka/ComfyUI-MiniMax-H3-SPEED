# ComfyUI MiniMax-H3 SPEED Sampler — V2


⚠️ **Noncommercial** — [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0) 
(I don't expect this to be used commercially. If it genuinely will be, message me.)



> *"Why make big noise when little noise do trick?"*

Make MiniMax-H3 video generation faster without retraining. SPEED starts denoising on a cheaper low-resolution grid, then increases resolution as finer detail becomes useful, avoiding full-resolution compute during the noisiest early steps.

> **MiniMax-H3 only.** Audio is always full-resolution.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

1. If your workflow already uses **SamplerCustomAdvanced**, replace it with **MiniMax H3 SPEED — Sampler** and wire the same `noise`, `guider`, `sigmas`, and `latent_image` inputs. If you use the basic all-in-one `KSampler`, first split it into ComfyUI's advanced sampling components so those inputs are available.
2. Set **`stages = 2`** or **`3`** (default) and hit Queue. The shipped Automatic calibration is the conservative Euler-derived 0.5% delta fit.


## Which node do I need?

**Automatic — Sampler**
Just `stages` (2, 3, or 4) that correspond to how many resolution stages there are. 
`2 = 0.5→1.0`, 
`3 = 0.33→0.66→1.0`, 
`4 = 0.25→0.5→0.75→1.0`. 

`Tolerance (Delta)`, `noise_amplitude`, and `noise_decay_exponent` determine where the resolution transitions happen. Leave them at the defaults unless you are using a Harvest calibration or deliberately experimenting.

**Manual — Sampler (Step-Through)**
You set up to four `(goal, resolution)` pairs yourself. For every stage except the last active one, `goal` is where that stage ends and `resolution` is its scale, such as `0.25` for quarter resolution. The last active stage always runs to the end of the sigma schedule, so its goal value is ignored. Active resolutions must increase and the final active resolution must be `1.0`. Set either value to `0` to skip a stage. Use Manual when copying a known schedule or testing a custom ladder.


**Sigma Harvest (Native Sampler)**
Run **once** with your current workflow to measure the selected native full-resolution sampler. Set Harvest to the same sampler, sigma schedule, step count, and `Tolerance (Delta)` you intend to use with SPEED. Then use the returned `sampler_name`, `delta`, `noise_amplitude` (A), and `noise_decay_exponent` (β) in the matching Automatic run. Harvest is still a native full-resolution pass; it does not run the SPEED chain.

Re-run Harvest when you materially change the checkpoint, sampler, LoRA/addons, scheduler, or step count.

For base H3 with **Euler**, you can use the shipped reference values instead of running Harvest:

- **Default (baked, 0.5%):** `Tolerance (Delta)=0.005, noise_amplitude=12.105, noise_decay_exponent=0.773` — `r² 0.70`
- **Balanced (1%):** `Tolerance (Delta)=0.01, noise_amplitude=12.436, noise_decay_exponent=0.786` — faster, with near-parity results in the reference clip

See the [evidence section](evidence/README.md) for what changes in generation.

## Supported samplers

SPEED supports exactly five samplers: **Euler**, **Heun**, **DPM2** (`dpm_2`),
**Exp Heun 2 X0** (`exp_heun_2_x0`), and **RES Multistep** (`res_multistep`).
The Automatic, Manual, and Sigma Harvest nodes expose the same list.

- **Euler** is the reference sampler and the default. The shipped calibration
  and benchmark evidence are Euler-derived. The SPEED boundary/alignment math
  itself is shared by all supported samplers.
- **Heun**, **DPM2**, and **Exp Heun 2 X0** are native stateless samplers.
  They can use extra model evaluations per step, which can reduce SPEED's
  wall-clock gain.
- **RES Multistep** is the only stateful sampler. Its adapter clears history at every resolution transition.

Automatic and Manual share the normal sampling inputs and return output and denoised LATENTs. Harvest shares the same `noise`, `guider`, `sigmas`, and `latent_image` inputs, plus sampler selection, and returns calibration JSON plus a diagnostic LATENT. Sigma Harvest stays native.

## Speed Improvements

Same 10s 0.5MP "world's most mediocre boss" office mug clip, same seed, corrected scheduler (post-PR-#37). Native Euler baseline: 571s. These measurements use Euler and the calibration values shown in the table. They do not establish parity for the other samplers.

| Fit | Mode | Time | Speedup | Quality |
|------|------|------|---------|---------|
| Δ0.005 `A12.105 β0.773` | 2-stage | 463s | 1.23× | visually near-native in this clip |
| Δ0.005 | 3-stage | 439s | 1.30× | visually near-native in this clip |
| Δ0.005 | 4-stage | 435s | 1.31× | near-native; mild melt artifact |
| Δ0.01 `A12.436 β0.786` | 2-stage | 450s | 1.27× | near parity |
| Δ0.01 | 3-stage | 410s | 1.39× | cleanest mug-landing beat |
| Δ0.01 | 4-stage | 384s | 1.49× | inconsistencies start appearing |
| Δ0.05 `A6.920 β0.766` | 2-stage | 278s | 2.05× | noticeable artifacting |
| Δ0.05 | 3-stage | 262s | 2.18× | very noticeable artifacting but still usable |
| Δ0.05 | 4-stage | 238s | 2.41× | intense artifacting and halo effect beginning |


See [evidence/README.md](evidence/README.md) for the full 10s GIFs (360p, 12fps).

**Rule of thumb:** quality-first use `stages 3` at Δ0.005; balanced use `stages 2` at Δ0.01; fast drafts use `stages 4` at Δ0.05.

## Troubleshooting

- **A transition falls outside the sigma schedule** → increase your scheduler step count or move Manual transition goals earlier. Every transition must happen after the first sigma and before the final sigma.
- **MiniMax-H3 sigma shifts are unavailable / the latent shape is rejected** → SPEED requires a real MiniMax-H3 video+audio latent and H3 sigma-shift metadata. It does not support SD, Flux, WAN, or other model families.
- **Text looks blurry / wobbly** → lower `Tolerance (Delta)` from `0.005` (0.5%) toward `0.001`, or use fewer stages. Both choices are more conservative and usually slower.
- **Prompt drifts / objects disappear on 4-stage** → too many hops. Drop to 2 or 3 stages.

## Current limitations

- Batch size is currently **1**.
- `noise_mask` / masked denoising is not supported.
- The main H3 `latent_image["samples"]` video and audio streams must start empty. SPEED's I2V support comes through MiniMax-H3 keyframe conditioning, which is resized and restored separately.
- SPEED expects the MiniMax-H3 nested latent layout: one video stream plus one audio stream.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

Sigma Harvest measures how the radial DCT power of the full-resolution **residual** `x - denoised` falls with frequency and fits `P(ω) = A·|ω|^-β`. The shipped Euler fit has β around 0.77. For each stage scale `s`, Automatic uses `ω = s·min(H,W)/2`, evaluates `P = A·ω^-β`, then computes `thr = 1/(1+√(δ/(P·(1+P-δ))))`. The first `sigmas[i] ≤ thr` becomes that stage boundary. The threshold is continuous; the actual boundary is quantized to your sigma schedule.

Re-calibrate with the Harvest node if you change checkpoint, sampler, or an addon that changes model behavior: wire `noise`, `guider`, `sigmas`, `latent_image`, and `Tolerance (Delta)`; run the selected native sampler at full resolution with the same sigma scheduler and step count you intend to use in SPEED; then select the same sampler in Automatic and copy `delta`, `noise_amplitude`, and `noise_decay_exponent`. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`.

**Noise policies:** `direct_coarse` is the default and fills newly exposed frequency bands from deterministic transition-seeded Gaussian noise. `coupled_full_grid` instead derives those bands from one seeded full-resolution Gaussian field, so every stage is coupled to the same full-grid realization. V2 keeps this mode for deterministic parity/ablation work, but there is currently no evidence that it is generally sharper or higher quality than `direct_coarse`.

With `direct_coarse`, `seed_offset` changes the deterministic high-frequency fill used at each resolution transition. It has no effect on `coupled_full_grid`, which takes those frequencies from the one full-resolution noise field. Leave it at 10000 unless you specifically want a different `direct_coarse` fill pattern for the same seed.

For the Manual node, `ratio_mode = steps` treats each goal as a global step index; `ratio_mode = ratio` treats it as a 0–1 fraction of the full denoising schedule.

</details>

## V2 major release

Updated the SPEED node and reworked many features.

The biggest change is **sampler support**. SPEED is no longer tied to Euler: V2 supports Euler, Heun, DPM2, Exp Heun 2 X0, and RES Multistep. The stateless samplers can use ComfyUI's normal sampler objects, but RES needs special handling because it remembers previous steps. That history is only valid while the latent grid stays the same, so V2 clears it whenever SPEED changes resolution instead of carrying stale state into a different-sized stage.

**Sigma Harvest is sampler-aware now** 

rewrote the **Automatic planning path** removing stale code and other issues within it.

Automatic and Manual now share the same planning code.

The **I2V handling** was also tightened up. Keyframe latents are always resized from their original full-resolution copy instead of repeatedly resizing an already-resized tensor. TLDR: I2V probably no longer gets progressively blurrier every time SPEED changes resolution.

Progress and previews now behave like **one generation**, not several unrelated sampler calls. 

**Sigma Harvest uses much less memory.** V1 kept every full-resolution residual tensor until the native pass was finished and only then analysed them. V2 converts each residual into its small radial power profile as soon as the callback receives it, then discards the large tensor.

I kept both noise policies for now. direct_coarse is still the default. I am not convinced coupled_full_grid is doing anything useful anymore, Kept it in but i dont know if it will improve anything. V2 at least avoids recomputing the same full-grid DCT at every transition.

See **[CHANGELOG.md](CHANGELOG.md)** for the detailed API-level release notes starting from when i started keeping track.

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
