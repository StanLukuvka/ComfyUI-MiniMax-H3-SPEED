# NEXT.md — ComfyUI-MiniMax-H3-SPEED (SPEED Sampler)

Last updated: 2026-08-24. Branch: `fix/harvest-delta-widget` (PR #24, draft).

## Status: shipped + documented, PR open for review

The pack installs and loads in ComfyUI (verified: 3 nodes registered —
Automatic, Manual Step-Through, Sigma Harvest). 54 tests pass, 5 skipped
(`speed_lab` sibling deselected). License is PolyForm Noncommercial 1.0.0.
README is quickstart-first with evidence GIFs.

## What shipped (since the reorg)

- **Three nodes** (flat files under `nodes/`):
  - `MiniMaxH3SPEEDSampler` (Automatic) — `stages` 2-4 even ladders
    (`2:0.5→1.0`, `3:0.33→0.66→1.0`, `4:0.25→0.5→0.75→1.0`), always
    `delta_custom` (steps auto from `Tolerance (Delta)` + `A/beta` via the
    power-spectrum threshold). Baked `Δ0.01 A7.394 β0.62`.
  - `MiniMaxH3SPEEDSamplerManual` (Manual Step-Through) — up to four
    `(goal, resolution)` pairs; `goal==0` or `resolution==0` disables that
    stage. `ratio_mode steps|ratio`.
  - `MiniMaxH3HarvestToConfig` (Sigma Harvest) — native Euler pass, residual
    capture per step, radial DCT power-law fit, flat `calibration` JSON to
    paste back into Automatic.
- `speed_scripts/` — `config.py` (SpeedConfig, presets, defaults),
  `flow.py` (sigma alignment), `h3_runtime.py` (multi-stage chain + Latent
  wiring), `harvest.py` (radial DCT + fit), `spectral.py` (DCT expansion),
  `latent.py` (Latent/RefLatent/LatentStage boundaries), `nodes_common.py`.
- `workflows/` — `video_minimax_h3_SPEED_SIGMA.json`,
  `video_minimax_h3_SPEED_Sigma_Calculated.json`,
  `video_minimax_h3_SPEED_Sigma_Manual.json`.
- `evidence/` — 10s 0.5MP Office mug sweep, 360p 12fps GIFs (committed),
  mp4s gitignored, `evidence/README.md` comparison page.

## CRITICAL: Sigma Harvest design rule (do NOT violate)

The harvest node (`MiniMaxH3HarvestToConfig`) is a **native Euler sampler
wrapper**, NOT a SPEED-chain wrapper. The SPEED sampler splices/re-aligns
sigmas at every stage boundary, which corrupts per-step sigma labels — you
CANNOT harvest a meaningful spectrum mid-SPEED-chain. The harvester runs ONE
full-res native Euler pass with a FIXED sigma schedule, captures
`residual = x - x0` at each step, fits `P = A*|ω|^(-β)`, emits flat
`calibration` JSON. Same inputs as any sampler node
(noise/guider/sigmas/latent_image). If you ever see the harvest node take a
`harvest_json` STRING input instead of native sampler inputs, it has regressed.

## CRITICAL: audio sigma-shift lookup (garbled sound regression)

`_active_av_shifts` in h3_runtime.py picks the H3 sigma shifts. Priority MUST
be: (1) `transformer_options["minimax_h3_sigma_shift_*"]`,
(2) `model.sigma_shift_*` (12.0 / 3.0 — AUTHORITATIVE),
(3) `diffusion_model.sigma_shift_*`, (4) LAST RESORT `model_sampling.shift`
(generic ComfyUI flow shift, often 1.0 — WRONG for H3). Commit 84e61ba
inverted this and garbled audio; the regression test
`test_av_shifts_ignore_generic_model_sampling_shift` locks the correct
priority. NEVER let generic `model_sampling.shift` shadow the H3-specific
`sigma_shift_video`/`audio`.

## CRITICAL: SPEED progress bar must stay ON

`run_repeated_stage_calls` defaults `disable_pbar=False` (bar VISIBLE). The
SPEED sampler node passes `disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED`.
Do NOT change the runtime default back to `True` — it hides the bar for every
run and is easy to miss.

## Open / deferred

- [ ] **Sigma-continuous stepper (novel research, out of scope until native
      SPEED).** Paper leaves it as exercise: sigma moves during video transit
      (`sigma_shift_video 12` vs audio 3) and during transition kappa
      realignment. Current pack precomputes discrete `transition_steps`
      (`first sigmas[i] ≤ thr`). Proposed: monitor live sigma per iteration
      and `spectral_expand` the instant `sigma ≤ thr_i` — ~10 lines in
      `h3_runtime`. See `NOTE_sigma_continuous_stepper.md` (deleted from
      repo, kept as email draft for SPEED researchers).
- [ ] Audio spectral expansion — no paper authority for audio `[B,C,2,T]`.
- [ ] `temporal_scales` UI exposure — config + runtime only, no node widget yet.
- [ ] Re-validate baked defaults on a fresh checkpoint (current `A7.394
      β0.62 δ0.01` from 0.6MP 40-step harvest; conservative `A12.454 β0.819
      δ0.005` also measured).
