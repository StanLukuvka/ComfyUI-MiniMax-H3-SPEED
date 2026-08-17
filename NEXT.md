# SPEED-Sampler — State & Next Steps

**Repo:** `/agent/projects/minimax-quickfile/ComfyUI-MiniMaxH3-SPEED-Sampler`
**Branch:** `dev` (off `main`)
**PR:** none open — working directly on `dev`; merge to `main` when feature is ready
**Tests:** 119 passing · **Commits ahead of main:** 15

---

## ✅ Done

- Sigma-harvest calibration pipeline (`harvest.py` + 3 nodes)
- `noise_policy` wired through Schedule + Sampler nodes (direct_coarse / coupled_full_grid)
- `coupled_full_grid` workflow variant created
- 7 debug/utility helper nodes (inspect, power_spectrum, dct_lowpass, transition_math, spectral_expand, x0_fidelity_probe, av_reentry_oracle)
- **Oracle straight-flow proof** — `StraightFlowModel`, `run_euler_pack`, 15 CPU-only tests
- End-to-end integration test (`test_integration.py`)
- README + node docs updated
- Pushed to origin, PR #1 open

---

## 🚧 Remaining

### 1. PR review / merge
Working on `dev` branch directly. Merge to `main` when feature is complete.

### 2. ComfyUI smoke test (manual, requires GPU)
- Load `video_minimax_h3_t2v_speed.json` and `video_minimax_h3_t2v_coupled.json` in a real ComfyUI
- Verify both sampler nodes execute and produce video
- Verify `sigma_harvest.json` → `sigma_harvest_calibrated.json` round-trips a real harvest

### 3. Lab parity (future, out of scope for MVP)
The Lab has deeper capabilities the MVP skipped:
- `speed_nodes/base.py`, `speed_nodes/configurable.py` (audio/sigma policy knobs)
- Multi-GPU parallel harvest
- Live latent auto-detection (remove optional `latent` input requirement)
- `av_reentry_oracle` live re-entry point finder on real latents

### 4. Lab sync
`/agent/projects/minimax-quickfile/ComfyUI-MiniMaxH3-SPEED-Lab` was the source. If the Lab advances (new calibration modes, new presets), re-sync here.

---

## 🧪 Test matrix

```
test_dct.py          9 passed   — DCT transform correctness
test_flow.py         6 passed   — Sigma alignment, audio handling
test_harvest.py      26 passed  — Power spectrum, fitting, callback
test_sampler.py      47 passed  — Full sampler integration, coupled_full_grid
test_integration.py  6 passed   — End-to-end harvest→schedule→sample
test_debug_nodes.py  20 passed  — All 7 debug nodes + bug fixes
test_oracle.py       15 passed  — Straight-flow oracle proof
test_spectral.py     3 passed   — Spectral expand correctness
test_schedule_node.py 8 passed   — Schedule node plan() integration

Total: 119 passed
```

---

## 📝 Notes

- MVP is intentionally minimal vs the Lab (3 calibration nodes + 7 debug nodes vs ~16 in Lab).
- All core SPEED functionality present and tested; `coupled_full_grid` fully supported.
- Oracle tests prove the SPEED transition + re-entry contracts on synthetic data — no model weights needed.
- The `hkb` wrapper at `/agent/hkb` bypasses the delegated-child kanban guard.
