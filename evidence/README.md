# SPEED Evidence — 10s 0.5MP Office Mug Clip

Same seed, `Δ0.01 A7.394 β0.62` unless noted. All at `960×544`, 24fps. Clips are 4s windows around the throw (2–6s) at 12fps, 480p — loop to see high-ω text and prompt.

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

Raw mp4s are in this folder (`Native.mp4`, `2 STAGE*.mp4`, etc.). See `TIMINGS.txt` for log. New harvest `Δ0.005 A12.45 β0.819 r²0.70 good` pending — may hold on `direct` without `coupled` tax.
