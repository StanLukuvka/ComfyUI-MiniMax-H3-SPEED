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

## Quick start

1. Load the example [`workflows/video_h3_SPEED.json`](workflows/video_h3_SPEED.json), or replace **SamplerCustomAdvanced** in your own workflow with **SPEED — Sampler** and connect the same `noise`, `guider`, `sampler`, `sigmas` and `latent_image` inputs. If you use the all-in-one `KSampler`, first split it into ComfyUI's advanced sampling nodes: **RandomNoise**, **CFGGuider** (or BasicGuider), **KSamplerSelect**, **BasicScheduler** and an empty latent.
2. Set **`model_type`**: `h3_video` for MiniMax-H3, `flow_image` for flow image models. It has no default.
3. Fill in **`noise_amplitude`** and **`noise_decay_exponent`**. They describe how your model, sampler and schedule behave, and SPEED will not run without them: left at 0, the node stops with an error.
   - MiniMax-H3 with `euler`: use the reference values `noise_amplitude = 35.4213`, `noise_decay_exponent = 1.0126`.
   - Anything else: run the **SPEED — Sigma Harvest** node once with the same sampler, schedule and size (see [Which node do I need?](#which-node-do-i-need) below), and copy its two numbers.
4. Set **`stages`** (3 is the default) and **`Tolerance (Delta)`** (0.005 for video, 0.05 for images), then queue.

The example workflows ship with these values already filled in.

## Which node do I need?

**SPEED — Sampler** (Automatic)  
Choose how many resolution stages you want:

`2 = 0.5→1.0`  
`3 = 0.33→0.66→1.0`  
`4 = 0.25→0.5→0.75→1.0`

`Tolerance (Delta)` (Δ in this README), `noise_amplitude` and `noise_decay_exponent` decide at which step each resolution switch happens. A smaller Δ switches later: safer, but it saves less time. `noise_amplitude` and `noise_decay_exponent` have no defaults: no model's calibration is baked in. Paste them from Harvest, or for H3 with Euler use the reference values below.

**SPEED — Sampler (Manual Step-Through)**  
For experimenting. Start with Automatic; use Manual when you want to pick the switch points yourself. Set up to four `(goal, resolution)` pairs. For every stage except the last active one, `goal` is the step where that stage ends (or a 0–1 fraction of the schedule with `ratio_mode = ratio`), and `resolution` is its scale, such as `0.25` for quarter resolution. The last active stage always runs to the end of the schedule, so its goal only needs to be non-zero.

Active resolutions must increase, and the last active resolution must be `1.0`. Set either value to `0` to skip a stage.

**SPEED — Sigma Harvest (Native Sampler)**  
Run Harvest with your current workflow to measure the connected sampler at full resolution. Use the same sampler, sigma schedule, step count, and `Tolerance (Delta)` that you intend to use with SPEED. Its `calibration` output is a JSON string whose `report` field holds the readable lines, including a ready-made paste line: connect it to a **Preview Any** node to read it. Then keep the same sampler connected to Automatic and copy the report's `Tolerance (Delta)`, `noise_amplitude` (A) and `noise_decay_exponent` (β) into it.

The report also gives `r²` and `health`, which say how well the measured curve fits. `good` or `fair` is usable. `weak` prints a warning: re-run or check your inputs before trusting it. `suspect` and `invalid` give no values to paste.

Harvest takes as long as one normal full-resolution render; it does not run the SPEED chain. Re-run it when you materially change the checkpoint, sampler, LoRA/addons, scheduler, or step count.

For base H3 with **Euler**, you can use the shipped reference values instead of running Harvest:

`noise_amplitude = 35.4213`, `noise_decay_exponent = 1.0126` (fit `r² 0.83`). Pick a `Tolerance (Delta)`:

- **Conservative:** `Tolerance (Delta)=0.005`.
- **Balanced:** `Tolerance (Delta)=0.01`, faster and still clean in the reference clip.
- **Fast:** `Tolerance (Delta)=0.05`, the fastest, and also clean in the reference clip. Check dark scenes and low step counts (turbo LoRA) for colour pulses.

See the [evidence section](evidence/h3_evidence/README.md) for examples of how these settings affect generation.

## Samplers

SPEED takes a normal ComfyUI `SAMPLER` input, so connect **KSamplerSelect** (or any other SAMPLER node). Each SPEED stage is its own sampler call, so multistep samplers start their history fresh at every resolution change. Euler is the reference: the shipped calibration and benchmark evidence are Euler-derived, so re-run Harvest with the sampler you use.

Tested on the 5 s mug clip (turbo LoRA, 8 steps, 2 stages, 0.2 MP on an L40S and 0.4 MP locally):

- **Same shot plan as Euler, sharpest:** `euler`, `heun`, `dpm_2`, `dpmpp_2m`, `res_multistep`, `uni_pc`, `ipndm`, `gradient_estimation`.
- **Work, softer and with different staging** (they inject noise every step): `euler_ancestral`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, `dpmpp_sde`, `er_sde`. `deis` also works but is softer.
- **Broken:** `lms` and `ipndm_v` (halos and smearing). They also break without SPEED on a few-step schedule.
- `heun`, `dpm_2` and `dpmpp_sde` run two model evaluations per step, so they take about 1.6× as long.

Automatic and Manual return two LATENTs: use `output`. `denoised_output` is the model's last clean-image estimate, for debugging. Harvest returns its calibration text plus a diagnostic LATENT.

## Example workflows

Each example workflow has Harvest, Automatic and Manual on one set of loaders, in Models, Conditioning, Sampling, SPEED and output groups. Only Automatic is active, already set to that model's harvested values.

`noise_policy` decides where the extra detail comes from at each switch. `coupled_full_grid` takes it from the same seeded noise a plain SamplerCustomAdvanced run uses, so it keeps that run's composition at the same seed. `direct_coarse` (the default) draws its own, so the picture differs from a plain run.

| Workflow | Model | `model_type` | Noise policy | Sampling | Automatic |
|---|---|---|---|---|---|
| [`video_h3_SPEED.json`](workflows/video_h3_SPEED.json) | MiniMax H3 | `h3_video` | `direct_coarse` | euler, simple, 20 steps | 3 stages, Δ 0.005, A 35.4213, β 1.0126 |
| [`image_moss_SPEED.json`](workflows/image_moss_SPEED.json) | MOSS Image | `flow_image` | `coupled_full_grid` | euler, sgm_uniform, 20 steps | 3 stages, Δ 0.05, A 48.4075, β 1.0333 |
| [`image_krea2_SPEED.json`](workflows/image_krea2_SPEED.json) | Krea 2 | `flow_image` | `coupled_full_grid` | euler, simple, 52 steps | 3 stages, Δ 0.05, A 9.7616, β 0.6308 |

## Image models

Set **`model_type = flow_image`** on the SPEED node (and on Harvest) to run a flow image model such as MOSS Image or Krea 2. Wire it like SamplerCustomAdvanced, with an empty latent (`EmptyLatentImage` works: SPEED fits it to the model's channels like SamplerCustomAdvanced does). `model_type` has no default; set it on every SPEED node. A value that does not match the model and latent stops with an error.

The image workflows' calibrations were harvested for their own prompt, seed and settings. For your model and settings, run Harvest with `model_type = flow_image` and paste its values in. At Δ 0.005 the harvested boundary lands in the first few steps of an image schedule, so it saves little; use **Δ 0.05**, or Manual with the 0.5× stage ending near half the steps.

**Image-to-image.** Image workflows usually use KSampler, whose `denoise` widget runs img2img. SPEED is wired like SamplerCustomAdvanced, so the same setting moves to **BasicScheduler**'s `denoise`, and the `latent_image` comes from **VAEEncode** instead of an empty latent. The first stage starts from the encoded image shrunk to that stage's size, noised to the schedule's first sigma, as SamplerCustomAdvanced does at full size. A lower `denoise` starts the schedule at a lower sigma. Automatic skips any coarse stage whose boundary sigma is already above it, so at low `denoise` it runs fewer stages and saves less; when it skips every coarse stage it runs as plain full-size sampling. Manual keeps the stages you set. Run Harvest on an empty latent at `denoise` 1, as for text-to-image: an encoded image changes the residuals it measures.

RTX 5070 Laptop, a ramen stall at night as a still ([evidence](evidence/image_evidence/README.md)); speedups use the unrounded times:

| Model | Native | Automatic 3 stages, Δ 0.05 | Manual, 0.5× to half the steps |
|---|---|---|---|
| MOSS Image, 1344×768, 20 steps, CFG 5 | 22 s | 18 s (1.26×) | 14 s (1.54×) |
| Krea 2 (raw, fp8), 1216×832, 52 steps, CFG 3.5 | 283 s | 215 s (1.31×) | 183 s (1.55×) |

Every SPEED run was clean and as detailed as native. With `noise_policy = coupled_full_grid`, SPEED draws its noise from the seed as SamplerCustomAdvanced does, so it keeps the native run's composition at the same seed; small details such as an object count can still change. `direct_coarse` draws its own coarse noise and gives a different picture.

## Speed improvements

A 5 s 0.5 MP clip of a latte-art pour in a dark café, same seed for every run, on a Modal L40S. The scene is dark, with small printed lettering and fine texture, so colour pulses and detail loss would show. All runs use one Harvest calibration, A 35.4213, β 1.0126. Native Euler took 276 s in the Δ 0.005 session and 288 s in the Δ 0.01/0.05 session; each speedup is against the native time from its own session.

These measurements use Euler. They do not establish parity for the other samplers.

| Δ | Mode | Time | Speedup |
|------|------|------|---------|
| 0.005 | 2-stage | 238s | 1.16× |
| 0.005 | 3-stage | 236s | 1.17× |
| 0.005 | 4-stage | 227s | 1.22× |
| 0.01 | 2-stage | 238s | 1.21× |
| 0.01 | 3-stage | 226s | 1.27× |
| 0.01 | 4-stage | 224s | 1.29× |
| 0.05 | 2-stage | 206s | 1.40× |
| 0.05 | 3-stage | 194s | 1.48× |
| 0.05 | 4-stage | 182s | 1.58× |

None of the ten clips has a colour flash. SPEED takes a different noise path, so the composition differs from native, but every run keeps the pour, the latte art and readable lettering. See [evidence/h3_evidence/README.md](evidence/h3_evidence/README.md) for the GIFs and links to the original MP4s.

**Rule of thumb:** use Δ 0.005 for quality-first runs, Δ 0.01 for a balance, and Δ 0.05 for the most speed. More stages are faster at every Δ.

## Troubleshooting

- **A transition falls outside the sigma schedule** → a resolution switch would land before the first step or after the last one. Use more steps, or move Manual goals inside the schedule.
- **"use model_type flow_image" / "use model_type h3_video"** → the `model_type` switch does not match the latent. H3 needs `h3_video`; plain image latents need `flow_image`.
- **MiniMax-H3 sigma shifts are unavailable** → `model_type h3_video` needs a real MiniMax-H3 model.
- **noise_amplitude and noise_decay_exponent are not set (0)** → the two values are still 0. Paste them from a Harvest run for this model, sampler and schedule, or use the H3 Euler reference values above.
- **Nodes marked deprecated** → workflows saved before the rename use the old `MiniMaxH3*` node IDs. They still run; right-click a node and replace it with the matching `SPEED` node when convenient.
- **Text looks blurry / wobbly** → lower Δ from `0.005` toward `0.001`, or use fewer stages. Both options are more conservative and usually slower.
- **Prompt drifts / objects disappear on 4-stage** → there may be too many resolution hops. Drop to 2 or 3 stages.

## Current limitations

- H3 batch size is currently **1**. Image models take any batch size.
- `model_type flow_image` takes single images only (`[B,C,H,W]`, or `[B,C,1,H,W]` for 3-dimensional latent formats such as Krea 2's Wan 2.1 format) and needs a flow (CONST) model, in Harvest too. Other models, such as EPS or v-prediction checkpoints, are rejected with an error.
- `noise_mask` / masked denoising is not supported.
- For `h3_video` the `latent_image` must start empty (both the video and audio streams). SPEED's I2V support comes through MiniMax-H3 keyframe conditioning, which is resized and restored separately.
- SPEED expects the MiniMax-H3 nested latent layout: one video stream plus one audio stream.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

Sigma Harvest measures how the radial DCT power of the full-resolution **residual** `x - denoised` falls with frequency and fits `P(ω) = A·|ω|^-β`. The shipped H3 Euler reference fit has β about 1.01.

For each stage scale `s`, Automatic uses `ω = s·min(H,W)/2`, evaluates `P = A·ω^-β`, then computes `thr = 1/(1+√(Δ/(P·(1+P-Δ))))`. The first `sigmas[i] ≤ thr` becomes that stage boundary. The threshold is continuous; the actual boundary is quantized to your sigma schedule.

Re-calibrate with the Harvest node if you change the checkpoint, sampler, or an addon that changes model behavior. Wire `noise`, `guider`, `sampler`, `sigmas`, `latent_image`, and `Tolerance (Delta)`; run the connected sampler at full resolution with the same sigma scheduler and step count you intend to use in SPEED; then connect the same sampler to Automatic and copy `Tolerance (Delta)`, `noise_amplitude` and `noise_decay_exponent`.

For base H3, the reference values were harvested at 20 steps with the `simple` scheduler, as in the example workflow.

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
