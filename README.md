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

1. Replace your `KSampler` / `SamplerCustomAdvanced` with **MiniMax H3 SPEED — Sampler (Automatic)**. Wire the same `noise`, `guider`, `sigmas`, `latent_image`.
2. Set **`stages = 2`** or **`3`** (default) and hit Queue. The shipped Automatic calibration is the conservative Euler-derived 0.5% delta fit.


## Which node do I need?

**Automatic — Sampler**
Just `stages` (2, 3, or 4) that correspond to how many resolution stages there are. 
`2 = 0.5→1.0`, 
`3 = 0.33→0.66→1.0`, 
`4 = 0.25→0.5→0.75→1.0`. 

`Tolerance (Delta)`, `noise_amplitude`, `noise_decay_exponent` determine at what steps each stage is triggered at, generally leave unless experimenting.

RES Multistep uses reset-only history. It clears previous-step history at every SPEED resolution transition.

**Manual — Sampler (Step-Through)**
You set up to four `(goal, resolution)` pairs yourself. `goal` = step where that stage ends, `resolution` = scale like `0.25` = quarter. Set `goal` or `resolution` to `0` to skip a stage. Use only to copy a paper schedule or test a custom ladder.


**Sigma Harvest (Native Sampler)**
Run **once** with your current workflow to measure your checkpoint with the selected native full-resolution sampler. It gives you sampler-specific `A / β` to paste into the matching Automatic run. Harvest is still a native full-resolution pass; it does not run the SPEED chain.
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
- **RES Multistep** is the only stateful sampler. Its adapter clears history at every resolution transition.

Automatic and Manual share the normal sampling inputs and return output and denoised LATENTs. Harvest shares the same `noise`, `guider`, `sigmas`, and `latent_image` inputs, plus sampler selection, and returns calibration JSON plus a diagnostic LATENT. Sigma Harvest stays native.

## Diagnostics

- **Sigma Harvest (Native Sampler)** runs one native full-res pass with the
  selected sampler and outputs a sampler-specific residual calibration (`A /
  β`) to paste into the matching Automatic configuration.

## Speed Improvements

Same 10s 0.5MP "world's most mediocre boss" office mug clip, same seed, corrected scheduler (post-PR-#37). Native Euler baseline: 571s. These baked measurements use Euler evidence and per-resolution harvest calibrations. They do not establish parity for the other samplers. The benchmark evidence below is Euler unless explicitly stated otherwise.

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


See [evidence/README.md](evidence/README.md) for full 10s GIFs (360p 12fps) and the review rubric.

**Rule of thumb:** quality-first use `stages 3` at Δ0.005; balanced use `stages 2` at Δ0.01; fast drafts use `stages 4` at Δ0.05.

## Troubleshooting

- **"Sigma schedule too short"** → increase `BasicScheduler` steps. The last stage boundary must leave at least one denoising step: with the final boundary at step `g`, you need ≥ `g + 2` sigmas (e.g. a 4-stage run with boundaries 3/5/8 needs ≥10 sigmas = 9 steps).
- **"H3 model required"** → this only works with a real MiniMax-H3 model (one that has `sigma_shift_video` / `sigma_shift_audio`). Not SD/Flux/WAN.
- **Text looks blurry / wobbly** → lower `Tolerance (Delta)` from `0.005` (0.5%) toward `0.001`, or use fewer stages. Both choices are more conservative and usually slower.
- **Prompt drifts / objects disappear on 4-stage** → too many hops. Drop to 2 or 3 stages.

## Advanced — you don't need this to use it

<details>
<summary>How Automatic picks the steps (click to expand)</summary>

It measures how noise power falls with frequency on a full-res run: `P(ω) = A·|ω|^-β` (β ~0.77 for MiniMax-H3's validated fits; see Defaults below). For each scale `s`, `ω = s·min(H,W)/2`, `P = A·ω^-β`, then `thr = 1/(1+√(δ/(P·(1+P-δ))))` (δ = Tolerance, 0.005 = 0.5% allowed error). The first `sigmas[i] ≤ thr` is where that stage ends. Continuous sigma, just quantized to your sigma schedule.

Re-calibrate with the Harvest node if you change checkpoint, sampler, or an addon that changes model behavior: wire `noise/guider/sigmas/latent + Tolerance`, run the selected native sampler at full resolution with the same sigma scheduler and step count you intend to use in SPEED, then copy its `calibration` JSON into the matching Automatic run's `noise_amplitude` / `noise_decay_exponent` / `Tolerance`. For base H3, the reference calibration workflow uses 28–32 steps with the `simple` sigma scheduler.

Stages are evenly spaced: `2: 0.5→1.0`, `3: 0.33→0.66→1.0`, `4: 0.25→0.5→0.75→1.0`.

**Noise policies:** `direct_coarse` is the default and fills newly exposed frequency bands from deterministic transition-seeded Gaussian noise. `coupled_full_grid` instead derives those bands from one seeded full-resolution Gaussian field, so every stage is coupled to the same full-grid realization. V2 keeps this mode for deterministic parity/ablation work, but there is currently no evidence that it is generally sharper or higher quality than `direct_coarse`.

`seed_offset` changes the per-stage high-frequency fill pattern — leave at 10000 unless you want a different fill pattern for the same seed. `ratio_mode steps` = goal is a step index, `ratio` = goal is a 0-1 fraction.

</details>

## V2 major release — what this PR changes

V2 is a major rewrite, not a tuning-only update. The three public ComfyUI node IDs stay the same, but the sampler/runtime architecture, calibration path, I2V lifecycle, and supported sampler surface have changed substantially.

This PR:

- expands generation from **Euler-only to five supported samplers**: Euler, Heun, DPM2, Exp Heun 2 X0, and RES Multistep;
- adds a **run-scoped deterministic RES Multistep adapter** that resets history at every SPEED resolution transition;
- makes **Sigma Harvest sampler-aware** while keeping it a native full-resolution calibration pass;
- rewrites Automatic planning to use the **live sigma schedule and live H3 latent geometry**, rather than cached dimensions or placeholder boundaries;
- consolidates Automatic and Manual schedule construction into one planner and keeps the runtime focused on execution;
- makes the **I2V keyframe lifecycle generation-local**, always resizing from pristine full-resolution conditioning and restoring it on completion or failure;
- keeps progress/preview on **one continuous run-wide timeline** across all SPEED stages;
- reduces Harvest residuals to CPU spectral profiles during the callback instead of retaining a full run of large residual tensors;
- precomputes `coupled_full_grid` spectral noise once per generation and reuses it across transitions;
- removes a large amount of legacy/configuration plumbing and adds broad regression coverage for the five-sampler matrix, RES state, I2V, global boundaries, spectral coupling, Harvest, and workflow compatibility.

### V1 → V2 compatibility

For normal ComfyUI use, migration should be small. Automatic and Manual still default to **Euler** when no sampler is specified, the existing inputs keep their order, and the same three node IDs remain registered. The main visible addition is the `sampler_name` selector.

If you import `speed_scripts` from Python directly, V2 does have internal API changes: cached `full_latent_h/full_latent_w` were removed from `SpeedConfig`, configs now require at least two stages, and the planner builders no longer take a latent just to cache its dimensions.

See **[CHANGELOG.md](CHANGELOG.md)** for the full release/migration notes.

## License

**PolyForm Noncommercial 1.0.0** — see [LICENSE.md](LICENSE.md). Noncommercial use only.
