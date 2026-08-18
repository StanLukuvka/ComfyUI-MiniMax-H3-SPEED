# Post-MVP Roadmap — MiniMax H3 SPEED Sampler

**Created:** 2026-08-18
**Branch:** `dev` (17 commits ahead of `main`)
**Tests:** 120 passing
**Paper:** arXiv:2605.18736 — Spectral Progressive Diffusion

---

## §0 Scope of this milestone

Every file touched or created:
- `minimax_h3_speed/spectral.py` — add temporal DCT axis
- `minimax_h3_speed/h3_runtime.py` — fix coupled-noise audio alignment
- `minimax_h3_speed/config.py` — add `temporal_scales` field
- `minimax_h3_speed/harvest.py` — no changes (already correct)
- `minimax_h3_speed/flow.py` — no changes (already correct)
- `minimax_h3_speed/oracle.py` — no changes
- `nodes/helper_nodes/dct_lowpass.py` — add temporal axis support
- `nodes/helper_nodes/spectral_expand.py` — add temporal axis support
- `sampler_node.py` — expose `temporal_scales` widget
- `nodes/helper_nodes/schedule.py` — expose `temporal_scales` widget
- `minimax_h3_speed/tests/test_spectral.py` — temporal DCT tests
- `minimax_h3_speed/tests/test_integration.py` — coupled-noise audio test
- `README.md` — document temporal scales

Deferred (§N Future milestones):
- Multi-GPU parallel harvest
- Live latent auto-detection (remove optional `latent` input)
- Fine-tuning recipe (LoRA on SPEED-expanded latents)

---

## Gap Audit

### ✅ Fully implemented and correct

| Paper component | Source | Tests |
|---|---|---|
| Eq. 8: Power spectrum P(ω) = A\|ω\|^-β | `h3_runtime.py:31` | test_harvest.py |
| Eq. 9: Activation time (Prop 1) | `h3_runtime.py:36` | test_harvest.py |
| Eq. 10: Per-resolution schedule via δ-optimal | `h3_runtime.py:116` | test_integration.py |
| Eq. 11: κ rescale (aligned_speed_sigma) | `flow.py:10` | test_flow.py |
| Eq. 12: Aligned timestep t̃ (time_shift_sigma) | `flow.py:21` | test_flow.py |
| Sec 4.1: Spectral noise expansion (2D DCT) | `spectral.py:104` | test_spectral.py |
| Sec 4.1: Coupled full-grid noise | `spectral.py:84` | test_spectral.py |
| Sec 4.2: Recover internal state | `flow.py:28` | test_flow.py |
| Sec 4.2: Clock re-index audio | `flow.py:52` | test_flow.py |
| Sec 4.2: Carry-preserving audio | `flow.py:37` | test_flow.py |
| Sec 4.2: Re-entry noise | `flow.py:76` | test_flow.py |
| Sigma harvest calibration | `harvest.py` | test_harvest.py (26 tests) |
| Oracle straight-flow proof | `oracle.py` | test_oracle.py (15 tests) |

### ❌ Gaps identified

#### GAP-1 [CORE] — Temporal DCT expansion missing

**Paper (Sec 4.1, Discussion):** SPEED operates on both spatial AND temporal frequency axes. The paper's DCT expansion is 3D for video (H, W, T), not 2D (H, W). The Discussion section explicitly notes temporal frequencies carry motion information and must be spectrally managed.

**Current:** `spectral.py` only does 2D DCT on spatial axes (H, W). `spectral_expand_dct` and `spectral_expand_dct_coupled` expand spatial dimensions only. The temporal axis T is left untouched, meaning spectral content along time is whatever random noise produces, not a controlled low-frequency band.

**Impact:** Video flicker / temporal inconsistency at resolution transitions. The spatial expansion is correct but temporal coherence breaks at every boundary.

**Fix:** Add 1D DCT along the temporal axis (dim=-3 for [B,C,T,H,W]).

```python
# spectral.py — new functions

def dct_temporal(value: torch.Tensor) -> torch.Tensor:
    """1D DCT along temporal axis (dim=-3 for [B,C,T,H,W])."""
    _validate_video_tensor(value)  # needs ndim >= 3
    work = value.float()
    T = work.shape[-3]
    basis = _basis(T, work.device)
    # transform along T: basis @ work (matmul on dim -3)
    work_t = work.transpose(-3, -2)  # [B,C,H,T,W]? no — wrong
    # Actually: apply along T by matmul
    # basis shape: [T, T], work shape: [..., T, H, W]
    # We want: result[..., t, h, w] = sum_t basis[t, t'] * work[..., t', h, w]
    # So: einsum or matmul with reshaping
    shape = work.shape
    work_2d = work.reshape(-1, T, shape[-2] * shape[-1])  # [N, T, HW]
    transformed = torch.matmul(basis, work_2d)  # [N, T, HW]
    return transformed.reshape(*shape)


def idct_temporal(coefficients: torch.Tensor) -> torch.Tensor:
    """Inverse 1D DCT along temporal axis."""
    work = coefficients.float()
    T = work.shape[-3]
    basis = _basis(T, work.device)
    shape = work.shape
    work_2d = work.reshape(-1, T, shape[-2] * shape[-1])
    restored = torch.matmul(basis.transpose(0, 1), work_2d)
    return restored.reshape(*shape)


def spectral_expand_dct_3d(
    value: torch.Tensor,       # [..., T_coarse, H_coarse, W_coarse]
    target_thw: tuple[int, int, int],
    sigma: float,
    seed: int,
) -> torch.Tensor:
    """3D spectral expansion: temporal + spatial DCT."""
    target_t, target_h, target_w = target_thw
    source_t, source_h, source_w = value.shape[-3:]
    # Generate full-res noise in spectral domain
    generator = torch.Generator(device=value.device)
    generator.manual_seed(int(seed))
    expanded = torch.randn(
        value.shape[:-3] + (target_t, target_h, target_w),
        generator=generator, device=value.device, dtype=torch.float32,
    )
    expanded.mul_(float(sigma))
    # Embed low-freq coefficients: temporal DCT of value, then spatial DCT
    source_coeffs = dct2(dct_temporal(value).transpose(-3, -2)).transpose(-3, -2)
    # ... or compose as dct_temporal then dct2 on spatial
    # Embed: expanded[..., :source_t, :source_h, :source_w] = source_coeffs
    # Then idct2 + idct_temporal
    ...
```

**Files:** `minimax_h3_speed/spectral.py`
**Authority:** Paper Sec 4.1, Discussion (temporal frequencies)
**Verification:** `python -m pytest minimax_h3_speed/tests/test_spectral.py -q --tb=short`
**Tests to add:**
- `test_dct_temporal_roundtrip` — `idct_temporal(dct_temporal(x)) == x` within 1e-5
- `test_dct_temporal_dc_component` — DC bin equals mean along T
- `test_spectral_expand_dct_3d_shape` — output shape matches target_thw
- `test_spectral_expand_dct_3d_lowfreq_preserved` — coarse coefficients preserved
- `test_spectral_expand_dct_3d_energy_bound` — expanded energy proportional to sigma²

---

#### GAP-2 [CORE] — Coupled-noise audio not DCT-expanded

**Paper (Sec 4.1):** When using coupled_full_grid noise, BOTH video and audio noise share the same full-grid spectral structure. The current code lowpasses video noise but passes audio noise through untouched.

**Current:** `h3_runtime.py:268-273` — coupled path DCT-expands video via `spectral_expand_dct_coupled` but audio goes through `reentry_noise(transitioned_audio, new_q)` without any spectral coupling. The audio noise is just the raw full-res noise.

**Impact:** Audio/video spectral coherence breaks at resolution transitions when using coupled mode. The paper's whole point is that coupled noise maintains spectral consistency across the video+audio bundle.

**Fix:** In the coupled path, audio should also carry the coupled spectral structure. Since audio is [B,C,2,T] (no spatial dims), this means temporal DCT expansion of audio noise.

```python
# h3_runtime.py — in the coupled_full_grid branch (line ~268)

if config.noise_policy == "coupled_full_grid":
    full_noise_video, full_noise_audio = _unpack_tensor(full_noise)
    expanded_video = spectral_expand_dct_coupled(
        internal_video,
        full_noise_video.to(device=internal_video.device, dtype=internal_video.dtype),
        q,
    )
    # FIX: also spectrally expand audio temporally
    expanded_audio = spectral_expand_audio_temporal(
        internal_audio,
        full_noise_audio,
        q,
    )
    transitioned_video = expanded_video * kappa
    transitioned_audio = expanded_audio  # already includes kappa via sigma
```

**Files:** `minimax_h3_speed/h3_runtime.py`, `minimax_h3_speed/spectral.py`
**Authority:** Paper Sec 4.1 (coupled noise), Discussion (audio spectral structure)
**Verification:** `python -m pytest minimax_h3_speed/tests/test_integration.py -q --tb=short -k coupled`
**Tests to add:**
- `test_coupled_audio_spectral_expansion` — audio is spectrally expanded, not raw noise
- `test_coupled_noise_preserves_audio_lowfreq` — low-freq audio content preserved across transition

---

#### ✅ GAP-3 [CORE] — Delta_custom not exposed in sampler node — **FIXED in f9930c7**

Was: `sampler_node.py:46` had `["explicit", "delta_custom"]` while `schedule.py:30` had `["manual_step", "manual_sigma", "delta_custom"]`. Two vocabularies for the same concept.

Now: sampler widget exposes the same three values as Schedule node. `MODE_TO_CONFIG` dict maps `manual_step`/`manual_sigma` → `explicit`, `delta_custom` passes through. Tested in `test_sampler_transition_mode_mapping`.

#### ✅ GAP-4 [QUALITY] — WAN 2.1 power-spectrum defaults — **FIXED in f0eb729**

Was: hardcoded WAN 2.1 values (A=219.48, β=2.42) caused delta_custom to fire at wrong sigmas on H3 latents when calibration is skipped.

Now: initial H3 estimates (A=150.0, β=2.0) with comment pointing to harvest pipeline. All node widgets + function defaults updated in sync.

#### ❌ GAP-5 [QUALITY] — Helper nodes use MockNested instead of real NestedTensor

**Current:** All 7 debug helper nodes (`dct_lowpass`, `spectral_expand`, `x0_fidelity_probe`, `av_reentry_oracle`, `power_spectrum`, `transition_math`, `inspect`) use a local `MockNested` class that mimics `NestedTensor.is_nested` / `.unbind()`. This works for testing but means these nodes won't work with real H3 latents in a live ComfyUI.

**Impact:** Debug nodes are useless in production. They exist for offline testing only.

**Fix:** Replace `MockNested` with a duck-typed wrapper that accepts either real `comfy.nested_tensor.NestedTensor` or any object with `is_nested` and `unbind()`. The nodes already check `is_nested` — just remove the MockNested construction and pass through whatever the upstream node provides.

```python
# nodes/helper_nodes/dct_lowpass.py — remove MockNested
# BEFORE:
class MockNested:
    def __init__(self, streams):
        self._streams = streams
        self.is_nested = True
    def unbind(self):
        return self._streams

# AFTER: just use the input directly
def execute(self, latent, cutoff=0.5):
    samples = latent.get("samples")
    if not getattr(samples, "is_nested", False):
        raise ValueError("Requires H3 nested latent")
    streams = list(samples.unbind())
    video = streams[0]
    ...
    return ({"samples": samples.__class__([filtered] + streams[1:])},)
    # Or: from comfy.nested_tensor import NestedTensor
    #      return ({"samples": NestedTensor([filtered] + streams[1:])},)
```

**Files:** All 7 helper node files in `nodes/helper_nodes/`
**Authority:** ComfyUI NestedTensor API
**Verification:** `python -m pytest minimax_h3_speed/tests/test_debug_nodes.py -q --tb=short`
**Tests to add:**
- `test_dct_lowpass_accepts_real_nested` — passes with object having is_nested=True + unbind()
- `test_spectral_expand_accepts_real_nested` — same pattern

---

#### GAP-6 [EXTENSION] — Temporal scale scheduling

**Paper (Sec 4.2):** SPEED supports independent scale schedules for temporal and spatial dimensions. Video latents can be expanded temporally (more frames) or spatially (higher resolution) at different rates.

**Current:** `config.py` has a single `scales` tuple that applies to spatial dims only. No temporal scaling — the coarse stage runs at full temporal resolution.

**Impact:** The "spectral" in SPEED includes temporal frequencies. Without temporal scaling, coarse stages process full-length video at low spatial resolution, missing the speedup from also reducing temporal resolution.

**Fix:** Add a `temporal_scales` field to SpeedConfig and thread it through the runtime.

```python
# config.py — new field
@dataclass(frozen=True)
class SpeedConfig:
    ...
    temporal_scales: tuple[float, ...] = ()  # empty = no temporal scaling

    def __post_init__(self):
        ...
        if self.temporal_scales:
            if len(self.temporal_scales) != len(self.scales):
                raise ValueError("temporal_scales must match scales length")
            if not all(0.0 < s <= 1.0 for s in self.temporal_scales):
                raise ValueError("temporal scales must be in (0, 1]")
```

```python
# h3_runtime.py — in run_repeated_stage_calls
stage_t = [
    max(1, round(full_t * ts)) for ts in config.temporal_scales
] if config.temporal_scales else [full_t] * n_stages
```

**Files:** `minimax_h3_speed/config.py`, `minimax_h3_speed/h3_runtime.py`, `sampler_node.py`, `nodes/helper_nodes/schedule.py`
**Authority:** Paper Sec 4.2 (per-resolution scheduling)
**Verification:** `python -m pytest minimax_h3_speed/tests/test_sampler.py minimax_h3_speed/tests/test_integration.py -q --tb=short`
**Tests to add:**
- `test_config_temporal_scales_validation` — mismatched lengths raise
- `test_runtime_with_temporal_scaling` — stage produces correct temporal dims
- `test_runtime_temporal_coarse_to_full` — final stage reaches full temporal res

---

## Task Blocks

### ✅ P5-001: [DONE] Unify transition_mode widgets (GAP-3)
**What:** Sampler node now accepts `["manual_step", "manual_sigma", "delta_custom"]` and maps to config-internal vocabulary.
**Files:** `sampler_node.py`, `minimax_h3_speed/tests/test_sampler.py`
**Verify:** `python -m pytest minimax_h3_speed/tests/test_sampler.py -q --tb=short`
**Status:** Committed in f9930c7. Test added.

### ✅ P5-002: [DONE] H3 power-spectrum defaults (GAP-4)
**What:** Replaced WAN 2.1 defaults (A=219.48, β=2.42) with H3 estimates (A=150.0, β=2.0).
**Files:** `minimax_h3_speed/config.py`, `sampler_node.py`, `nodes/helper_nodes/schedule.py`
**Verify:** `python -m pytest minimax_h3_speed/tests/ -q --tb=short`
**Status:** Committed in f0eb729.

### P5-003: [TODO] Temporal DCT expansion (GAP-1)
**What:** Add `dct_temporal`, `idct_temporal`, `spectral_expand_dct_3d` to `spectral.py`. Wire into `h3_runtime.py` runtime for 3D spectral expansion at boundaries.
**Files:** `minimax_h3_speed/spectral.py`, `minimax_h3_speed/h3_runtime.py`
**Authority:** Paper Sec 4.1, Discussion
**Verify:** `python -m pytest minimax_h3_speed/tests/test_spectral.py -q --tb=short`

### P5-004: [TODO] Coupled-noise audio spectral expansion (GAP-2)
**What:** In the `coupled_full_grid` branch of `h3_runtime.py`, DCT-expand audio noise temporally alongside video. Add `spectral_expand_audio_temporal` to `spectral.py`.
**Files:** `minimax_h3_speed/h3_runtime.py`, `minimax_h3_speed/spectral.py`
**Authority:** Paper Sec 4.1
**Verify:** `python -m pytest minimax_h3_speed/tests/test_integration.py -q --tb=short -k coupled`

### P5-005: [TODO] Remove MockNested from helper nodes (GAP-5)
**What:** Replace MockNested construction with pass-through NestedTensor usage in all 7 helper nodes. Test with duck-typed NestedTensor.
**Files:** `nodes/helper_nodes/dct_lowpass.py`, `nodes/helper_nodes/spectral_expand.py`, `nodes/helper_nodes/x0_fidelity_probe.py`, `nodes/helper_nodes/av_reentry_oracle.py`, `nodes/helper_nodes/power_spectrum.py`, `nodes/helper_nodes/transition_math.py`, `nodes/helper_nodes/inspect.py`
**Authority:** ComfyUI NestedTensor API
**Verify:** `python -m pytest minimax_h3_speed/tests/test_debug_nodes.py -q --tb=short`

### P5-006: [TODO] Temporal scale scheduling (GAP-6)
**What:** Add `temporal_scales` to SpeedConfig, thread through runtime, expose in sampler/schedule nodes.
**Files:** `minimax_h3_speed/config.py`, `minimax_h3_speed/h3_runtime.py`, `sampler_node.py`, `nodes/helper_nodes/schedule.py`
**Authority:** Paper Sec 4.2
**Verify:** `python -m pytest minimax_h3_speed/tests/test_sampler.py minimax_h3_speed/tests/test_integration.py -q --tb=short`

---

## §N Future milestones (deferred)

1. **Multi-GPU parallel harvest** — distribute harvest runs across GPUs for faster calibration
2. **Live latent auto-detection** — remove optional `latent` input by probing model for H3 geometry
3. **Fine-tuning recipe** — LoRA training on SPEED-expanded latents (paper Sec 4.3)
4. **Lab sync** — re-import from `ComfyUI-MiniMaxH3-SPEED-Lab` if it advances
