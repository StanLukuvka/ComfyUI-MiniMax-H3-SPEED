# SPEED Evidence: Image Models (`model_type = flow_image`)

One still per model: a ramen stall on a Tokyo side street at night, with a readable sign, a cook, two customers and exactly six paper lanterns, the third one blue. The sign, the count and the one blue lantern make composition changes easy to see. Each prompt follows its model's prompting study (MOSS: plain prose, one sentence per element, no quality prefix; Krea 2: quality prefix, medium first, text clause, key concepts repeated at the end). The workflows are [`workflows/image_moss_SPEED.json`](../../workflows/image_moss_SPEED.json) and [`workflows/image_krea2_SPEED.json`](../../workflows/image_krea2_SPEED.json).

**Machine:** RTX 5070 Laptop (8 GB) + 30 GB RAM, ComfyUI 0.41.0. Times are wall-clock per queued prompt with the models already loaded and the prompt encoded (sampling + VAE decode). Not comparable to the Modal L40S numbers in [h3_evidence](../h3_evidence/README.md).

The `direct_coarse` rows draw their transition fill on the CPU, so they reproduce on any device.

**Seeds and noise.** Every SPEED run uses `noise_policy = coupled_full_grid` unless marked otherwise. That policy draws the full-resolution noise from the seed exactly as SamplerCustomAdvanced does, low-passes it for the coarse stage and reuses its bands at each transition. So **plain at the same seed shares SPEED's noise field**, and that plain run is the reference image. `seed_offset` only seeds the transition fill under `direct_coarse`; `coupled_full_grid` ignores it. A plain run at seed + 10000 is listed too: it shares no noise with SPEED and shows how much a different draw moves the picture.

## MOSS-Image

`mossImage_sft_pruned_int8`, 1344×768, 20 steps, CFG 5, euler / sgm_uniform, SLG (uncond skip, layer 22), empty negative, seed 1. Harvest (Δ 0.005): `noise_amplitude 48.4075`, `noise_decay_exponent 1.0333`, r² 0.90 (good).

<details><summary>Prompt</summary>

```
A candid photograph of a small ramen stall on a narrow Tokyo side street at night. An elderly cook with short grey hair, a white headband and a navy apron ladles broth into a bowl behind the wooden counter. Steam rises from three large silver pots on the stove behind him. A hand-painted wooden sign above the stall reads "RAMEN 400" in white letters. Exactly six plain paper lanterns with no writing on them hang in a row along the eave. Five of the lanterns are red, and the third lantern from the left is blue. Two customers sit on wooden stools at the counter with their backs to the camera. Wet cobblestones in the foreground reflect the lantern light. Medium-wide shot at eye level, with the cook centred and his face in sharp focus.
```
</details>

| Mode | Time | Speedup | Lanterns (blue at) | Notes |
|------|------|---------|-------|-------|
| Plain, seed 1 | 22 s | 1× | 7 (3rd) | 22.3 / 22.2 s over two runs; the reference |
| Plain, seed 10001 | 22 s | — | 6 (3rd) | different noise, different picture |
| Harvest | 30 s | — | — | one full-resolution pass |
| Automatic, 2 stages, Δ 0.005 | 21 s | 1.07× | 8 (4th) | 0.5× stage ends at step 2 |
| Automatic, 2 stages, Δ 0.05 | 18 s | 1.21× | 6 (3rd) | |
| Automatic, 3 stages, Δ 0.05 | 18 s | 1.26× | 8 (3rd and 7th) | 17.6 / 17.7 / 17.4 s over three runs; two blue lanterns |
| Automatic, 4 stages, Δ 0.05 | 18 s | 1.27× | 7 (2nd) | |
| Manual, 0.5× to step 10 of 20 | 14 s | **1.54×** | 6 (5th) | 14.4 / 14.7 / 14.2 s over three runs |
| Automatic, 3 stages, Δ 0.05, `direct_coarse` | 17 s | 1.30× | 8 (5th) | different noise, different picture |

| Plain 22 s | Automatic 3 stages Δ 0.05, 18 s | Manual to step 10, 14 s |
|---|---|---|
| ![MOSS plain](images/moss_plain.jpg) | ![MOSS auto 3](images/moss_auto3.jpg) | ![MOSS manual](images/moss_manual.jpg) |

On MOSS, SPEED keeps plain's scene (stall, cook, sign, wet street) but re-frames it: the cook moves to the centre and the customers to the ends of the counter. The lantern row is where it varies most.

## Krea 2

`krea2Raw_v10_full_fp8`, 1216×832, 52 steps, CFG 3.5, euler / simple, empty negative, seed 291. Harvest (Δ 0.005): `noise_amplitude 9.7616`, `noise_decay_exponent 0.6308`, r² 0.63 (fair). At Δ 0.005 the 0.5× boundary lands at step 7 of 52.

<details><summary>Prompt</summary>

```
masterpiece, best quality, candid 35mm film photograph, real person, natural skin texture: a small ramen stall on a narrow Tokyo side street at night. An elderly cook with short grey hair, a white headband and a navy apron ladles broth into a bowl behind the wooden counter, with steam rising from three large silver pots behind him. The wooden sign above the stall reads "RAMEN 400" in large, clear letters, and there is no other text. Exactly six plain paper lanterns hang in a row along the eave: five are red, and the third lantern from the left is blue. Two customers sit on wooden stools at the counter with their backs to the camera, and wet cobblestones in the foreground reflect the lantern light. Medium-wide shot at eye level. ramen stall, six paper lanterns, one blue lantern, elderly cook
```
</details>

| Mode | Time | Speedup | Lanterns (blue at) | Notes |
|------|------|---------|-------|-------|
| Plain, seed 291 | — | — | 6 (4th) | the reference image; not timed |
| Plain, seed 10291 | 283 s | 1× | 7 (3rd) | the timing baseline; a lantern covers half the sign |
| Harvest | 273 s | — | — | one full-resolution pass |
| Automatic, 2 stages, Δ 0.005 | 271 s | 1.04× | 6 (4th) | almost the same picture as plain |
| Automatic, 2 stages, Δ 0.05 | 226 s | 1.25× | 8 (5th) | |
| Automatic, 3 stages, Δ 0.05 | 215 s | 1.31× | 7 (4th) | |
| Automatic, 4 stages, Δ 0.05 | 252 s | 1.12× | 7 (4th) | |
| Manual, 0.5× to step 26 of 52 | 183 s | **1.55×** | 8 (5th) | |
| Automatic, 3 stages, Δ 0.05, `direct_coarse` | 255 s | 1.11× | 6 (3rd) | a different picture: sign between the lanterns, customers on the right |

Each Krea 2 row is a single run. The 13 GB model streams through the 8 GB GPU, so times swing by tens of seconds: Automatic 3 stages took 215 s with `coupled_full_grid` and 255 s with `direct_coarse` on the same schedule. Read the Automatic speedups as rough; Manual's 1.55× matches MOSS.

| Plain 283 s | Automatic 3 stages Δ 0.05, 215 s | Manual to step 26, 183 s |
|---|---|---|
| ![Krea 2 plain](images/krea2_plain.jpg) | ![Krea 2 auto 3](images/krea2_auto3.jpg) | ![Krea 2 manual](images/krea2_manual.jpg) |

## What this shows

- With `coupled_full_grid`, SPEED keeps plain's composition at the same seed, closely on Krea 2 and more loosely on MOSS. `direct_coarse` draws different coarse noise and gives a different picture.
- Every run was clean (no seams, banding or colour shifts) and kept the sign legible.
- What moves is small, countable detail: the lantern count and where the blue one hangs. Neither model reliably draws exactly six lanterns, even plain, so judge SPEED against its own plain run, not against the prompt.
- At Δ 0.005 the low-resolution stage ends in the first few steps, so it saves 4–7%. **Δ 0.05**, or Manual with the 0.5× stage ending at half the steps (about 1.55× on both models), is the useful setting for images.
- Fixed per-run costs such as the VAE decode are a larger share of a still than of a 10 s video, so image speedups stay below the H3 ones.
