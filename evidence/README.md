# SPEED Evidence — 10s 0.5MP Office Mug Clip

Same seed, same prompt ("World's Most Mediocre Boss" office mockumentary) — 1× RTX 5080 + 128GB RAM unless noted. All at `960×544`, 24fps, 10.125s. Full 10s GIFs at 360p 12fps

Native is full-res Euler (no SPEED): **571.49s**.

## Δ=0.005, noise_amplitude=12.105, noise_decay_exponent=0.773

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| Native | 571.49s | 1× | ![Native](gifs/NATIVE.gif) |
| 2-stage 0.5→1.0 | 462.96s | 1.23× | ![0.005 2-stage](gifs/0.005_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 438.85s | 1.30× | ![0.005 3-stage](gifs/0.005_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 435.21s | 1.31× | ![0.005 4-stage](gifs/0.005_4_stage.gif) |


## Δ0.010, noise_amplitude=12.436, noise_decay_exponent=0.786

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| 2-stage 0.5→1.0 | 450.38s | 1.27× | ![0.01 2-stage](gifs/0.01_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 409.83s | 1.39× | ![0.01 3-stage](gifs/0.01_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 384.13s | 1.49× | ![0.01 4-stage](gifs/0.01_4_stage.gif) |


## Δ0.050, noise_amplitude=6.920, noise_decay_exponent=0.766

| Mode | Time | Speedup | GIF |
|------|------|---------|-----|
| 2-stage 0.5→1.0 | 278.44s | 2.05× | ![0.05 2-stage](gifs/0.05_2_stage.gif) |
| 3-stage 0.33→0.66→1.0 | 262.32s | 2.18× | ![0.05 3-stage](gifs/0.05_3_stage.gif) |
| 4-stage 0.25→0.5→0.75→1.0 | 237.57s | 2.41× | ![0.05 4-stage](gifs/0.05_4_stage.gif) |
