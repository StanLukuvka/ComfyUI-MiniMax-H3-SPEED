# SPEED Evidence — 10s 0.5MP Office Mug Clip

Same seed, `Δ0.01 A7.394 β0.62` — 1× RTX 5080 + 128GB RAM unless noted. All at `960×544`, 24fps. Full 10s clips at 12fps, 360p (embedded) — loop to see high-ω text and prompt.

Native is full-res Euler (no SPEED). `direct` = `direct_coarse`, `coupled` = `coupled_full_grid`.

| Mode | Time | GIF |
|------|------|-----|
| Native | 832.83s (cold) | ![Native](gifs/Native.gif) |
| 2-stage `direct` 0.5→1.0 | 651.29s cold | ![2 STAGE](gifs/2%20STAGE.gif) |
| 2-stage `coupled` 0.5→1.0 | 608.56s | ![2 coupled](gifs/2%20STAGE%20coupled%20full%20grid.gif) |
| 3-stage `direct` 0.33→0.66→1.0 | 415.97s | ![3 STAGE](gifs/3%20STAGE.gif) |
| 3-stage `coupled` 0.33→0.66→1.0 | 616.16s | ![3 coupled](gifs/3%20stage%20coupled%20full.gif) |
| 4-stage `direct` 0.25→0.5→0.75→1.0 | 261.64s | ![4 STAGE](gifs/4%20STAGE.gif) |
| 4-stage `coupled` 0.25→0.5→0.75→1.0 | 608.49s | ![4 coupled](gifs/4%20stage%20coupled%20full%20grid.gif) |

Alt fit `Δ0.005 A12.45 β0.819 r²0.70 good` (more conservative, still baked as `0.01` by default):

| Mode (Δ0.005) | Time | GIF |
|------|------|-----|
| 2-stage `direct` 0.5→1.0 | 671.87s | ![2 0.005](gifs/delta%200.005%202%20stage.gif) |
| 3-stage `direct` 0.33→0.66→1.0 | 540s | ![3 0.005](gifs/delta%200.005%203%20stage.gif) |
| 4-stage `direct` 0.25→0.5→0.75→1.0 | 400s | ![4 0.005](gifs/delta%200.005%204%20stage.gif) |

Paste `Tolerance=0.005, noise_amplitude=12.454, noise_decay_exponent=0.819` into Automatic to try the alt fit.

Raw mp4s are in this folder. See `TIMINGS.txt` for log.
