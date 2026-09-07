# FLOW-PRODUCED
# SPEED Sigma Harvest (Continuous) — measurement reference

`SPEED Sigma Harvest (Continuous)` is the node
`MiniMax H3 SPEED — SPEED Sigma Harvest (Continuous)`
(`nodes/sampler_speed_sigma_harvest_node.py`). It runs one real multi-stage
SPEED generation and records the model's spectral state at every denoising
step. It is observational only: it does not move stage transitions during
generation.

This document defines what the node measures, what the output JSON contains,
how to read it, and what it cannot tell you.

## Three different diagnostics

The pack has three diagnostics with different jobs. They are not interchangeable:

| Node | Generation | Measurement |
|---|---|---|
| Sigma Harvest (`MiniMaxH3HarvestToConfig`) | native full-res Euler | one aggregate residual calibration (`A` / `beta`) to paste into Automatic |
| Sigma Trace (`MiniMaxH3SigmaTrace`) | native full-res Euler | per-step x0 telemetry with normalized low/mid/high bands |
| SPEED Sigma Harvest (Continuous) | real multi-stage SPEED | per-step absolute spectra, stage-aware, both x0 and residual |

Sigma Trace's bands are normalized to sum to 1. Normalized proportions are
useful telemetry, but they discard absolute signal strength, which the SPEED
activation equation needs. The continuous harvester measures absolute radial
DCT power.

## The two measurement bases

Every measured step records two spectra under separate JSON namespaces:

- `x0_signal` — the DCT power spectrum of the model's `denoised` prediction.
  This is the paper's `P_omega = E[|x0^(omega)|^2]` on clean data. It is the
  theoretically relevant quantity for the delta-optimal activation equation.
- `residual` — the DCT power spectrum of `x_current - x0`. This is the same
  basis the existing native Sigma Harvest calibrates on. Its per-step `A` /
  `beta` values are directly comparable to that calibration.

Both names carry the fitted coefficients (`x0_signal.fit.A`,
`residual.fit.A`, ...), so no number in the JSON is ambiguous about which
spectrum it came from.

The baked Automatic defaults (`noise_amplitude=7.394`,
`noise_decay_exponent=0.62`) come from the residual basis. They are an
empirical H3 residual calibration, not the paper's clean-data power spectrum.
The continuous harvester never seeds its x0 statistics from those coefficients.

## Output JSON schema

Top level (v1 schema):

```json
{
  "schema_version": 1,
  "type": "minimax_h3_speed_sigma_harvest",
  "mode": "observational",
  "adaptive_control": false,
  "measurement_bases": ["denoised_x0_video", "residual_x_minus_x0_video"],
  "sampler": "euler",
  "static_scheduler": {
    "stages": 4,
    "scales": [0.25, 0.5, 0.75, 1.0],
    "delta": 0.01,
    "noise_amplitude": 7.394,
    "noise_decay_exponent": 0.62,
    "resolved_transition_steps": [3, 5, 8],
    "original_sigmas": []
  },
  "analysis": {
    "stride": 1,
    "smoothing_alpha": 0.25,
    "boundary_band_half_width": 1.0,
    "fit_omega_min": 0.5,
    "fit_omega_max_policy": "half_min_current_stage",
    "measurement_mode": "both",
    "store_radial_profiles": false
  },
  "records": [],
  "transitions": [],
  "summary": {}
}
```

`measurement_mode` is `both`, `x0_only`, or `residual_only`. When
`store_radial_profiles` is off (default), records contain fitted/scalar
telemetry only; with it on, each record also carries the 1D radial profile
(small, CPU-side), which is useful for research plots at the cost of JSON size.

Each `records[]` entry (per measured callback):

```json
{
  "callback_index": 6,
  "global_schedule_index": 6,
  "stage_index": 1,
  "stage_scale": 0.5,
  "stage_local_step": 2,
  "video_shape": [1, 24, 31, 23, 40],
  "sigma": {
    "actual": 0.621,
    "actual_next": 0.58,
    "original": 0.59,
    "original_next": 0.55
  },
  "x0_signal": {
    "fit": {
      "status": "ok",
      "A": 201.3,
      "beta": 2.11,
      "r_squared": 0.91,
      "n_bins": 18,
      "health": "good"
    },
    "current_boundary": {
      "transition_index": 1,
      "omega": 11.25,
      "direct_available": true,
      "power_point": 1.42,
      "power_band_mean": 1.39,
      "activation_threshold_point": 0.632,
      "activation_threshold_band": 0.629,
      "eligible_point": true,
      "eligible_band": true,
      "ema_power": 1.35,
      "ema_threshold": 0.625,
      "eligible_ema": true
    },
    "fit_predictions": [
      {
        "transition_index": 1,
        "omega": 11.25,
        "P": 1.37,
        "threshold": 0.627,
        "predicted_original_step": 6,
        "measurement": "power_law_fit"
      }
    ]
  },
  "residual": {
    "fit": {
      "status": "ok",
      "A": 7.8,
      "beta": 0.67,
      "r_squared": 0.63,
      "n_bins": 18,
      "health": "fair"
    }
  }
}
```

Sigma values: after a transition aligns the working schedule
(`working_sigmas[boundary] = aligned_sigma`), the next stage starts at an
`actual` sigma that differs from the `original` input sigma. `actual` answers
"what timestep is this stage really operating at"; `original` answers "where
is this in the user's original scheduler trajectory". Both are recorded.

Boundary power: `power_point` is the radial DCT power interpolated at the
current stage's next transition frequency
(`omega = stage_scale * min(full_H, full_W) / 2` — the frequency canonical
SPEED uses for that transition). `power_band_mean` is the mean over a narrow
radial band of half-width `boundary_band_half_width` around it. The point
value is closest to the theory; the band value is a less noisy empirical
signal. Both are stored; neither is silently substituted for the other.

Direct vs extrapolated: the current stage can only directly observe
frequencies its own resolution represents. `direct_available: true` marks a
direct measurement; anything derived for future stages comes from the fitted
power law and is labelled `measurement: "fit_extrapolation"`. Never read an
extrapolated prediction as an observation.

Eligibility and EMA: `eligible_point` / `eligible_band` / `eligible_ema`
answer, per step, whether the live x0 spectrum would already make the stage
eligible to expand if it were controlling SPEED
(`actual_sigma <= threshold`). `ema_power` is a log-space exponential moving
average of the raw boundary power, seeded from the first valid x0 measurement
of the run. It is recorded for study only.

Transitions: each `transitions[]` entry records one real resolution change
(`from_stage`, `to_stage`, `global_schedule_index`, scales, ratio,
`sigma_before_alignment`, `sigma_after_alignment`, `kappa`, source/target
shapes). Coincident transitions each emit their own event even when the
intermediate stage runs zero denoising steps.

Summary: per non-final stage — callback counts, the static planned transition
step, the first live-eligible steps (direct raw / EMA / fit) with their
difference from the static step, and first/last/min/max/mean of x0 and
residual `A` / `beta` plus mean `r_squared`. Summaries operate on scalars;
radial profiles are never averaged across stage resolutions.

## How to interpret a run

The experiment answers three questions:

1. Does the residual fit move significantly during generation? If
   `A_residual` / `beta_residual` are nearly constant across a run, the
   static native Sigma Harvest calibration may already be sufficient.
2. Does x0 boundary power move even when `A` / `beta` stay stable? If yes,
   the global two-parameter fit is hiding useful local spectral behavior and
   direct `P(omega)` measurement matters.
3. Do live thresholds move transition timing by whole scheduler steps? A
   threshold that drifts slightly is irrelevant if it quantizes to the same
   static step every run. What matters is a consistent, repeatable gap, such
   as static step 7 vs live step 5 across runs.

Practical reading order per run: check `summary` first for the
static-vs-live step gaps, then plot the per-step series (below) to see
whether the signal is stable across seeds and prompts.

## Plot suggestions

For each run (2/3/4-stage, several seeds and prompts):

- `A` / `beta` / `r_squared` vs global step, one series for `x0_signal.fit`
  and one for `residual.fit`.
- `current_boundary.power_point` and `current_boundary.ema_power` vs global
  step (direct boundary power, raw and smoothed).
- Static threshold vs live thresholds (direct point, band, EMA, fit) vs
  global step.
- Vertical markers at each entry of `transitions` (actual resolution changes).

## Known limitations

- Direct boundary measurement only exists at frequencies the current stage
  can represent. Future-stage values are fit extrapolations, labelled as
  such in the JSON.
- The x0/residual EMA is observational. It never feeds back into the sampler.
- The node cannot adapt transitions mid-stage: the whole stage sigma slice is
  handed to `guider.sample()` before its callbacks run, so a "transition now"
  measurement cannot stop a running stage. Same-run adaptive scheduling needs
  a larger runtime rewrite (own the Euler loop) and is out of scope here.
- Intermediate options exist if adaptive scheduling is later justified:
  a one-`guider.sample()`-per-step prototype (simple, slow), or a two-pass
  approach (harvest once, derive a schedule, rerun SPEED with it as a fixed
  schedule). Owning the Euler loop in the SPEED runtime is the preferred
  long-term architecture.
- Spectral fits consume `P(omega)` through a two-parameter power law. If the
  real spectrum bends, the fit-derived threshold diverges from the direct
  measurement — that gap is exactly why both are recorded.

## Old benchmark evidence

3/4-stage evidence collected before the global transition fix (PR #37) must
be rerun before it is compared against anything. Under the old scheduler the
stages did not run where the configuration said they did, so telemetry from
that period is contaminated. The continuous harvester must only be interpreted
on the corrected scheduler.
