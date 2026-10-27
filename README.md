# ComfyUI MiniMax-H3 SPEED Sampler — V2

⚠️ **Noncommercial** — [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0)  
(I don't expect this to be used commercially. If it genuinely will be, message me.)

> *"Why make big noise when little noise do trick?"*

Make MiniMax-H3 video generation faster without retraining. SPEED starts denoising on a cheaper low-resolution grid, then increases the resolution as finer detail becomes useful. This avoids paying for full-resolution compute during the noisiest early steps.

> **MiniMax-H3 video, plus flow image models** (tested on MOSS Image and Krea 2) through the `model_type` switch. H3 audio always stays at full resolution.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

1. If your workflow already uses **SamplerCustomAdvanced**, replace it with **SPEED — Sampler** and connect the same `noise`, `guider`, `sampler`, `sigmas`, and `latent_image` inputs (`sampler` comes from **KSamplerSelect** or any other SAMPLER node). If you use the basic all-in-one `KSampler`, first split it into ComfyUI's advanced sampling components so those inputs are available.
2. Set **`model_type`** to your model (`h3_video` or `flow_image`); it has no default. Paste `noise_amplitude` and `noise_decay_exponent` from a Harvest run, or for H3 with Euler use the reference values under Sigma Harvest below; they start at 0, which stops with an error.
3. Set **`stages = 2`** or **`3`** (default), then queue the workflow.

Or start from the example: [`workflows/video_h3_SPEED.json`](workflows/video_h3_SPEED.json) has Harvest, Automatic and Manual on one set of H3 loaders, in Models, Conditioning, Sampling, SPEED and output groups; only Automatic is active.

## Which node do I need?

**Automatic — Sampler**  
Choose how many resolution stages you want:

`2 = 0.5→1.0`  
`3 = 0.33→0.66→1.0`  
`4 = 0.25→0.5→0.75→1.0`

`Tolerance (Delta)`, `noise_amplitude`, and `noise_decay_exponent` control when those resolution transitions happen. `noise_amplitude` and `noise_decay_exponent` have no defaults: no model's calibration is baked in. Paste them from Harvest, or for H3 with Euler use the reference values below.

**Manual — Sampler (Step-Through)**  
Set up to four `(goal, resolution)` pairs yourself. For every stage except the last active one, `goal` is where that stage ends and `resolution` is its scale, such as `0.25` for quarter resolution. The final active stage always runs to the end of the sigma schedule, so its goal value is ignored.

Active resolutions must increase, and the final active resolution must be `1.0`. Set either value to `0` to skip a stage. Use Manual when copying a known schedule or testing a custom ladder.

**Sigma Harvest (Native Sampler)**  
Run Harvest with your current workflow to measure the connected sampler at full resolution. Use the same sampler, sigma schedule, step count, and `Tolerance (Delta)` that you intend to use with SPEED. Then keep the same sampler connected to Automatic and copy the returned `delta`, `noise_amplitude` (A), and `noise_decay_exponent` (β) into it.

Harvest is still a native full-resolution pass; it does not run the SPEED chain. Re-run it when you materially change the checkpoint, sampler, LoRA/addons, scheduler, or step count.

For base H3 with **Euler**, you can use the shipped reference values instead of running Harvest:

- **Conservative (0.5%):** `Tolerance (Delta)=0.005, noise_amplitude=12.105, noise_decay_exponent=0.773` — `r² 0.70`
- **Balanced (1%):** `Tolerance (Delta)=0.01, noise_amplitude=12.436, noise_decay_exponent=0.786` — faster, with near-parity results in the reference clip

See the [evidence section](evidence/h3_evidence/README.md) for examples of how these settings affect generation.

## Samplers

SPEED takes a normal ComfyUI `SAMPLER` input, so connect **KSamplerSelect** (or any other SAMPLER node). Each SPEED stage is its own sampler call, so multistep samplers start their history fresh at every resolution change. Euler is the reference: the shipped calibration and benchmark evidence are Euler-derived, so re-run Harvest with the sampler you use.

Tested on the 5 s mug clip (turbo LoRA, 8 steps, 2 stages, 0.2 MP on an L40S and 0.4 MP locally):

- **Same shot plan as Euler, sharpest:** `euler`, `heun`, `dpm_2`, `dpmpp_2m`, `res_multistep`, `uni_pc`, `ipndm`, `gradient_estimation`.
- **Work, softer and with different staging** (they inject noise every step): `euler_ancestral`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `dpmpp_sde`, `er_sde`. `deis` also works but is softer.
- **Broken:** `lms` and `ipndm_v` (halos and smearing). They also break without SPEED on a few-step schedule.
- `heun`, `dpm_2` and `dpmpp_sde` run two model evaluations per step, so they take about 1.6× as long.

Automatic and Manual return output and denoised LATENTs; Harvest returns calibration JSON plus a diagnostic LATENT.

## Image models

Set **`model_type = flow_image`** on the SPEED node (and on Harvest) to run a flow image model such as MOSS Image or Krea 2. Wire it like SamplerCustomAdvanced, with an empty latent (`EmptyLatentImage` works: SPEED fits it to the model's channels like SamplerCustomAdvanced does). `model_type` has no default; set it on every SPEED node. A value that does not match the model and latent stops with an error.

Example workflows: [`workflows/image_moss_SPEED.json`](workflows/image_moss_SPEED.json) and [`workflows/image_krea2_SPEED.json`](workflows/image_krea2_SPEED.json). Each has Harvest, Automatic and Manual on one set of loaders, in Models, Conditioning, Sampling, SPEED and output groups; only Automatic is active, already set to that model's harvested values and `noise_policy = coupled_full_grid`.

There is no shipped calibration for image models. Run Harvest with `model_type = flow_image` on your model and settings, and paste its values in. At Δ 0.005 the harvested boundary lands in the first few steps of an image schedule, so it saves little; use **Δ 0.05**, or Manual with the 0.5× stage ending near half the steps.

RTX 5070 Laptop, a ramen stall at night as a still ([evidence](evidence/image_evidence/README.md)):

| Model | Plain | Automatic 3 stages, Δ 0.05 | Manual, 0.5× to half the steps |
|---|---|---|---|
| MOSS Image, 1344×768, 20 steps, CFG 5 | 22 s | 18 s (1.26×) | 14 s (1.54×) |
| Krea 2 (raw, fp8), 1216×832, 52 steps, CFG 3.5 | 283 s | 215 s (1.31×) | 183 s (1.55×) |

Every SPEED run was clean and as detailed as plain. With `noise_policy = coupled_full_grid`, SPEED draws its noise from the seed as SamplerCustomAdvanced does, so it keeps the plain run's composition at the same seed; small details such as an object count can still change. `direct_coarse` draws its own coarse noise and gives a different picture.

## Speed improvements

Same 10s 0.5 MP "world's most mediocre boss" office mug clip, same seed, corrected scheduler (post-PR-#37). Native Euler baseline: 571s.

These measurements use Euler and the calibration values shown in the table. They do not establish parity for the other samplers.

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

See [evidence/h3_evidence/README.md](evidence/h3_evidence/README.md) for the full 10s GIFs (360p, 12fps).

**Rule of thumb:** for quality-first runs, use `stages 3` at Δ0.005; for a balance of speed and quality, use `stages 2` at Δ0.01; for fast drafts, use `stages 4` at Δ0.05.

## Troubleshooting

- **A transition falls outside the sigma schedule** → increase your scheduler step count or move Manual transition goals earlier. Every transition must happen after the first sigma and before the final sigma.
- **"use model_type flow_image" / "use model_type h3_video"** → the `model_type` switch does not match the latent. H3 needs `h3_video`; plain image latents need `flow_image`.
- **MiniMax-H3 sigma shifts are unavailable** → `model_type h3_video` needs a real MiniMax-H3 model.
- **noise_amplitude and noise_decay_exponent are not set (0)** → paste the values from a Harvest run for this model, sampler and schedule.
- **Nodes marked deprecated** → workflows saved before the rename use the old `MiniMaxH3*` node IDs. They still run; right-click a node and replace it with the matching `SPEED` node when convenient.
- **Text looks blurry / wobbly** → lower `Tolerance (Delta)` from `0.005` (0.5%) toward `0.001`, or use fewer stages. Both options are more conservative and usually slower.
- **Prompt drifts / objects disappear on 4-stage** → there may be too many resolution hops. Drop to 2 or 3 stages.

## Current limitations

- H3 batch size is currently **1**. Image models take any batch size.
- `model_type flow_image` takes single images only (`[B,C,H,W]`, or `[B,C,1,H,W]` for 3-dimensional latent formats such as Krea 2's Wan 2.1 format) and needs a flow (CONST) model, in Harvest too. Other models, such as EPS or v-prediction checkpoints, are rejected with an error. img2img with a non-empty latent is not supported.
- `noise_mask` / masked denoising is not supported.
- The `latent_image` must start empty (for H3, both the video and audio streams). SPEED's I2V support comes through MiniMax-H3 keyframe conditioning, which is resized and restored separately.
- SPEED expects the MiniMax-H3 nested latent layout: one video stream plus one audio stream.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

Sigma Harvest measures how the radial DCT power of the full-resolution **residual** `x - denoised` falls with frequency and fits `P(ω) = A·|ω|^-β`. The shipped Euler fit has β around 0.77.

For each stage scale `s`, Automatic uses `ω = s·min(H,W)/2`, evaluates `P = A·ω^-β`, then computes `thr = 1/(1+√(δ/(P·(1+P-δ))))`. The first `sigmas[i] ≤ thr` becomes that stage boundary. The threshold is continuous; the actual boundary is quantized to your sigma schedule.

Re-calibrate with the Harvest node if you change the checkpoint, sampler, or an addon that changes model behavior. Wire `noise`, `guider`, `sampler`, `sigmas`, `latent_image`, and `Tolerance (Delta)`; run the connected sampler at full resolution with the same sigma scheduler and step count you intend to use in SPEED; then connect the same sampler to Automatic and copy `delta`, `noise_amplitude`, and `noise_decay_exponent`.

For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`.

**Noise policies:** `direct_coarse` is the default and fills newly exposed frequency bands with deterministic transition-seeded Gaussian noise. `coupled_full_grid` instead derives those bands from one seeded full-resolution Gaussian field, coupling every stage to the same full-grid realization. I have not confirmed that `coupled_full_grid` improves quality, so `direct_coarse` remains the default.

With `direct_coarse`, `seed_offset` changes the deterministic high-frequency fill used at each resolution transition. It has no effect on `coupled_full_grid`, which takes those frequencies from the one full-resolution noise field. Leave it at 10000 unless you specifically want a different `direct_coarse` fill pattern for the same seed.

For the Manual node, `ratio_mode = steps` treats each goal as a global step index; `ratio_mode = ratio` treats it as a 0–1 fraction of the full denoising schedule.

</details>

## V2 major release

V2 updates the SPEED nodes and reworks several parts of the implementation.

The biggest change is **sampler support**. SPEED is no longer tied to Euler: every node takes a normal ComfyUI `SAMPLER` input, the same as SamplerCustomAdvanced. See [Samplers](#samplers) for what was tested.

**Sigma Harvest is now sampler-aware**, so its calibration reflects the sampler you are actually using.

I also rewrote the **Automatic planning path**, removing stale code and simplifying how stage transitions are calculated.

Automatic and Manual now share the same planning code instead of maintaining separate versions of the same logic.

The **I2V handling** was tightened up as well. Keyframe latents are always resized from their original full-resolution copy instead of repeatedly resizing an already-resized tensor. TL;DR: I2V probably no longer gets progressively blurrier every time SPEED changes resolution.

Progress and previews now behave like **one generation** instead of several unrelated sampler calls.

**Sigma Harvest uses much less memory.** V1 kept every full-resolution residual tensor until the native pass finished and only analysed them afterwards. V2 converts each residual into a small radial power profile as soon as the callback receives it, then discards the large tensor.

I kept both noise policies for now. `direct_coarse` is still the default. I am not convinced `coupled_full_grid` is doing anything useful anymore, but I also cannot confirm that removing it would not make some cases worse, so I left it in. V2 at least avoids recomputing the same full-grid DCT at every transition.

See **[CHANGELOG.md](CHANGELOG.md)** for the detailed API-level changes from the point where I started tracking them.

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
