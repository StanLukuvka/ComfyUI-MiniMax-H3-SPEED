# ComfyUI MiniMax-H3 SPEED Sampler

<!-- FLOW-PRODUCED: RES V3 hybrid boundary documentation. -->

⚠️ **Noncommercial** — [LICENSE.md](LICENSE.md) (PolyForm Noncommercial 1.0.0) 
(I don't expect this to be used commercially. If it genuinely will be, message me.)



> *"Why make big noise when little noise do trick?"*

Make MiniMax-H3 video faster without re-training. 
Starts the denoise at low resolution, then upsamples to full resolution when finetuned detail starts appearing within noise. Allowing us to save on generations.

> **MiniMax-H3 only.** Audio is always full-resolution.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/StanLukuvka/ComfyUI-MiniMax-H3-SPEED.git
# restart ComfyUI
```

1. Replace your `KSampler` / `SamplerCustomAdvanced` with **MiniMax H3 SPEED — Sampler (Automatic)**. Wire the same `noise`, `guider`, `sigmas`, `latent_image`.
2. Set **`stages = 2`** or **`3`** (balanced, default) and hit Queue. Current default settings are the conservative sigma harvest at 0.5% delta.


## Which node do I need?

**Automatic — Sampler**
Just `stages` (2, 3, or 4) that correspond to how many resolution stages there are. 
`2 = 0.5→1.0`, 
`3 = 0.33→0.66→1.0`, 
`4 = 0.25→0.5→0.75→1.0`. 

`Tolerance (Delta)`, `noise_amplitude`, `noise_decay_exponent` determine at what steps each stage is triggered at, generally leave unless experimenting.

`res_history_mode` only affects RES Multistep. `reset` is the default: it
discards RES previous-step history at every SPEED resolution transition, so
the first later RES interval rebuilds history from the new resolution.
`projected` is the historical negative control: it DCT-projects the previous
denoised history into the target video geometry and rebases its sigma
metadata. The GPU A/B on the known aggressive workload reproduced transient
flash / colour artifacts with `projected` while `reset` stayed stable. This is
external V2 evidence, not GPU validation supplied by this repository.
`hybrid` is experimental: on the first interval after a spatial-only
transition it blends a VIDEO DCT block from a second-order candidate into the
first-order update and uses the first-order update for audio in V3.0.
For flat host tensors, hybrid requires recorded target video/audio shapes and
fails closed when that metadata is missing or inconsistent instead of guessing
the stream split.
`hybrid` is an experiment, not a recommendation: it is not established as
correct or valid RES, and the one-interval blend is not second-order accurate;
`reset` remains the default.

**Manual — Sampler (Step-Through)**
You set up to four `(goal, resolution)` pairs yourself. `goal` = step where that stage ends, `resolution` = scale like `0.25` = quarter. Set `goal` or `resolution` to `0` to skip a stage. Use only to copy a paper schedule or test a custom ladder.

Manual SPEED exposes the same RES history choices: `reset` (default),
`projected` (historical negative control — external V2 GPU evidence reproduced
transient artifacts with it), and `hybrid` (experimental one-interval VIDEO
DCT blend, first-order audio in V3.0; not a recommendation).


**Sigma Harvest (Native Sampler)**
Run **once** with your current workflow to measure your checkpoint with the selected native full-resolution sampler. It gives you sampler-specific `A / β` to paste into the matching Automatic run.
If you are using LoRAs, or other models, addons, or optimisations that change how the model behaves, I recommend running it to ensure it is tuned to your specific workload.

You can instead use the following values for base H3:

- **Default (baked, 0.5%):** `Tolerance (Delta)=0.005, noise_amplitude=12.105, noise_decay_exponent=0.773` — `r² 0.70`
- **Balanced (1%):** `Tolerance (Delta)=0.01, noise_amplitude=12.436, noise_decay_exponent=0.786` — near parity, faster

See the [evidence section](evidence/README.md) for what changes in generation.

## Supported samplers

SPEED supports exactly five samplers: **Euler**, **Heun**, **DPM2** (`dpm_2`),
**Exp Heun 2 X0** (`exp_heun_2_x0`), and **RES Multistep** (`res_multistep`).
The Automatic, Manual, and Sigma Harvest nodes expose the same list.

- **Euler** is the reference sampler and the default. SPEED's baked evidence
  and kappa boundary alignment were derived from Euler measurements.
- **Heun**, **DPM2**, and **Exp Heun 2 X0** are native stateless samplers.
  They can use extra model evaluations per step, which can reduce SPEED's
  wall-clock gain.
- **RES Multistep** is the only stateful sampler. Its SPEED adapter exposes three
  boundary policies: `reset` is the default and cold-starts RES history after
  each resolution transition; `projected` reproduces the historical
  project-and-rebase behavior as a negative control — external V2 GPU evidence
  on the known aggressive workload reproduced transient flash / colour
  artifacts with it while `reset` stayed stable; `hybrid` is an experimental
  one-interval VIDEO DCT blend between first- and second-order candidates
  with first-order audio in V3.0. Neither `projected` nor `hybrid` claims
  native RES continuity across a geometry change: native RES does not define
  a geometry-changing boundary, and `hybrid` in particular is not established
  as correct or valid RES and is not second-order accurate. The supported
  adapter is deterministic and non-ancestral only: no SDE and no CFG++.
- All five names are public, but RES remains experimental until a user passes
  the H3 GPU validation gate. The repository provides automated seam and
  state tests, not a claim of measured hardware parity or native ComfyUI
  validation for the hybrid path.

Automatic and Manual share the normal sampling inputs and return output and denoised LATENTs. Harvest shares the same `noise`, `guider`, `sigmas`, and `latent_image` inputs, plus sampler selection, and returns calibration JSON plus a diagnostic LATENT. Sigma Harvest stays native and has no `res_history_mode` widget.

## Diagnostics

- **Sigma Harvest (Native Sampler)** runs one native full-res pass with the
  selected sampler and outputs a sampler-specific residual calibration (`A /
  β`) to paste into the matching Automatic configuration.

## Speed Improvements

Same 10s 0.5MP "world's most mediocre boss" office mug clip, same seed, corrected scheduler (post-PR-#37). Native Euler baseline: 571s. These baked measurements use Euler evidence and per-resolution harvest calibrations. They do not establish parity for the other samplers. The benchmark evidence below is Euler unless explicitly stated otherwise.

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

Re-calibrate with the Harvest node if you change checkpoint, sampler, or an addon that changes model behavior: wire `noise/guider/sigmas/latent + Tolerance`, run the selected native sampler at full resolution with the same sigma scheduler and step count you intend to use in SPEED, then copy its `calibration` JSON into the matching Automatic run's `noise_amplitude` / `noise_decay_exponent` / `Tolerance`. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`.

`seed_offset` changes the per-stage high-frequency fill pattern — leave at 10000 unless you want a different fill pattern for the same seed. `ratio_mode steps` = goal is a step index, `ratio` = goal is a 0-1 fraction.

</details>

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
