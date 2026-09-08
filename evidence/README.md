# SPEED Evidence — 10s 0.5MP Office Mug Clip

Same seed, same prompt ("World's Most Mediocre Boss" office mockumentary) — 1× RTX 5080 + 128GB RAM unless noted. All at `960×544`, 24fps, 10.125s. Full 10s GIFs at 360p 12fps (embedded) — loop to read the mug text and see the story beats.

All SPEED runs below were generated on the **corrected multi-stage scheduler** (global transition indexing, PR #37). Older evidence predating that fix is invalid — 3/4-stage runs transitioned late.

Native is full-res Euler (no SPEED): **571.49s**.

Story beats to check: mug toss near window (~50%), window-blinds melt (~62.5% — a native-model artifact), mug on forehead (~75%), man on back with mug landing on the floor (~87.5%).

Per-resolution calibrations were used (harvest at the render resolution):

| Fit | Paste into Automatic | Native-parity verdict |
|-----|----------------------|----------------------|
| Δ0.005 | `Tolerance=0.005, A=12.105, β=0.773` | quality-safe: matches or beats native detail |
| Δ0.01 | `Tolerance=0.010, A=12.436, β=0.786` | near parity, faster |
| Δ0.05 | `Tolerance=0.050, A=6.920, β=0.766` | budget tier: visible transition pops, all beats + text survive |

## Δ0.005 — quality-safe (7/10 each, native 6)

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| Native | 571.49s | 1× | ![Native](gifs/NATIVE.gif) |
| 2-stage 0.5→1.0 | 462.96s | 1.23× | ![0.005 2-stage](gifs/0.005_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 438.85s | 1.30× | ![0.005 3-stage](gifs/0.005_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 435.21s | 1.31× | ![0.005 4-stage](gifs/0.005_4_stage.gif) |

All three beat native on mug-text legibility and final-beat clarity; the blinds-melt artifact is milder than native. 3-stage has the best text, 2-stage the cleanest mug-landing beat.

## Δ0.01 — balanced (7.5–8.5/10)

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| 2-stage 0.5→1.0 | 450.38s | 1.27× | ![0.01 2-stage](gifs/0.01_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 409.83s | 1.39× | ![0.01 3-stage](gifs/0.01_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 384.13s | 1.49× | ![0.01 4-stage](gifs/0.01_4_stage.gif) |

Near native parity. Sharpest toss (2-stage), cleanest mug-landing beat (3-stage), no texture artifacts found in any run. The native blinds-melt artifact does not appear in any SPEED render — the multi-stage pipeline suppresses it.

## Δ0.05 — budget (6.5–7.5/10)

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| 2-stage 0.5→1.0 | 278.44s | 2.05× | ![0.05 2-stage](gifs/0.05_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 262.32s | 2.18× | ![0.05 3-stage](gifs/0.05_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 237.57s | 2.41× | ![0.05 4-stage](gifs/0.05_4_stage.gif) |

No failure boundary: mug text stays correct, faces hold, all 4 beats present in every run. Degradation concentrates in transition pops (~the blinds beat and the fall) and motion-section ghosting. 4-stage is the most temporally stable of the class; fine for previz/drafts, not final delivery.

## Notes

- `times.txt` has the raw timings.
- Review method: contact sheets per video (every 12th frame), scored on mug text, face integrity, texture artifacts, story-beat coherence, temporal stability.
- The old (pre-PR-#37) 3/4-stage evidence showed compounding transition drift — garbled text, face doubling, splatter. None of those signatures reproduce on the corrected scheduler.
