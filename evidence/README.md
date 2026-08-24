# SPEED Evidence — 10s 0.5MP Office Mug Clip

Same seed, `Δ0.01 A7.394 β0.62` unless noted. All at `960×544`, 24fps. Full 10s clips at 12fps, 360p (embedded) — loop to see high-ω text and prompt.

Native is full-res Euler (no SPEED). `direct` = `direct_coarse`, `coupled` = `coupled_full_grid`.

| Mode | Time | GIF | Notes |
|------|------|-----|-------|
| Native | 832.83s (cold) | ![Native](gifs/Native.gif) | reference — full-res |
| 2-stage `direct` 0.5→1.0 | 651.29s cold | ![2 STAGE](gifs/2%20STAGE.gif) | good, prompt intact |
| 2-stage `coupled` 0.5→1.0 | 608.56s | ![2 coupled](gifs/2%20STAGE%20coupled%20full%20grid.gif) | slightly sharper, `Mediocre` correct |
| 3-stage `direct` 0.33→0.66→1.0 | 415.97s | ![3 STAGE](gifs/3%20STAGE.gif) | **blurry** — notably destroys |
| 3-stage `coupled` 0.33→0.66→1.0 | 616.16s | ![3 coupled](gifs/3%20stage%20coupled%20full.gif) | **restored** — sharp, but slower than direct |
| 4-stage `direct` 0.25→0.5→0.75→1.0 | 261.64s | ![4 STAGE](gifs/4%20STAGE.gif) | unusable — heavy blur |
| 4-stage `coupled` 0.25→0.5→0.75→1.0 | 608.49s | ![4 coupled](gifs/4%20stage%20coupled%20full%20grid.gif) | sharp but prompt drifts |

Alt fit `Δ0.005 A12.45 β0.819 r²0.70 good` (more conservative, still baked as `0.01` by default):

| Mode (Δ0.005) | Time | GIF | Notes |
|------|------|-----|-------|
| 2-stage `direct` 0.5→1.0 | 671.87s | ![2 0.005](gifs/delta%200.005%202%20stage.gif) | slower than 0.01 `coupled`, very clean |
| 3-stage `direct` 0.33→0.66→1.0 | 540s | ![3 0.005](gifs/delta%200.005%203%20stage.gif) | **really good**, borderline hireq |
| 4-stage `direct` 0.25→0.5→0.75→1.0 | 400s | ![4 0.005](gifs/delta%200.005%204%20stage.gif) | usable but halo (sigma quantization) |

Paste `Tolerance=0.005, noise_amplitude=12.454, noise_decay_exponent=0.819` into Automatic to try it. 4-stage halo = discrete `thr→step` vs continuous sigma (see note).

Raw mp4s are in this folder (`Native.mp4`, `2 STAGE*.mp4`, etc.). See `TIMINGS.txt` for log. New harvest `Δ0.005 A12.45 β0.819 r²0.70 good` pending — may hold on `direct` without `coupled` tax.
