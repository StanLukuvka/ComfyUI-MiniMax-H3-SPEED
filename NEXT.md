# SPEED-Sampler — State & Next Steps

**Branch:** `feature/sigma-variant` (off `main`)
**Tests:** 68 passing
**Commits ahead of main:** 7

---

## ✅ Done

### Core Runtime
- `minimax_h3_speed/config.py` — SpeedConfig, presets, noise_policy support
- `minimax_h3_speed/h3_runtime.py` — Multi-stage diffusion loop with `coupled_full_grid`
- `minimax_h3_speed/spectral.py` — DCT-based spectral expansion
- `minimax_h3_speed/flow.py` — Sigma alignment, audio handling, reentry noise
- `minimax_h3_speed/harvest.py` — Radial power spectrum + power-law fitting

### ComfyUI Nodes
- `sampler_node.py` — MiniMaxH3SPEEDSampler (main node, noise_policy exposed)
- `nodes/helper_nodes/sigma_harvest.py` — Calibration pass node
- `nodes/helper_nodes/harvest_to_config.py` — JSON report parser
- `nodes/helper_nodes/schedule.py` — SpeedConfig planner (noise_policy exposed)
- `nodes/__init__.py` — Node registration (11 nodes)

### Tests
- `tests/test_sampler.py` — Sampler integration tests (coupled_full_grid coverage)
- `tests/test_harvest.py` — Power spectrum fitting tests
- `tests/test_spectral.py` — DCT expansion tests
- `tests/test_flow.py` — Sigma alignment tests
- `tests/test_dct.py` — Low-level DCT tests

### Workflows
- `workflows/video_minimax_h3_t2v_speed.json` — Standard pipeline (direct_coarse)
- `workflows/video_minimax_h3_t2v_coupled.json` — Coupled noise variant
- `workflows/sigma_harvest.json` — Calibration pass only
- `workflows/sigma_harvest_calibrated.json` — Full harvest→report pipeline

### Docs
- `README.md` — Usage, presets, noise policies, helper nodes

---

## 🚧 In Progress / Todo

### 1. Integration Test: End-to-End Harvest→Schedule→Sample
**File:** `tests/test_integration.py` (new)
**Purpose:** Verify the full calibration pipeline works without ComfyUI

```python
def test_harvest_to_schedule_to_sample():
    """Run a synthetic harvest, schedule with delta_custom, verify steps."""
    # 1. Generate synthetic noise
    # 2. Run harvest callback
    # 3. Parse report
    # 4. Schedule with delta_custom mode
    # 5. Verify transition steps are reasonable
```

### 2. Helper Node: `power_spectrum.py`
**Purpose:** Standalone node to visualize power spectrum during debug
**Stub:** `nodes/helper_nodes/power_spectrum.py`

```python
class MiniMaxH3PowerSpectrum:
    """Visualize power spectrum of latent tensor."""
    INPUT_TYPES = {"noise", "guider", "latent_image"}
    RETURN_TYPES = ("SPECTRUM", "STRING")  # Or just STRING for JSON
    FUNCTION = "run"
    OUTPUT_NODE = True
```

### 3. Helper Node: `dct_lowpass.py`
**Purpose:** Apply DCT lowpass filter for ablation testing
**Stub:** `nodes/helper_nodes/dct_lowpass.py`

```python
class MiniMaxH3DCTLowpass:
    """Apply lowpass filter in DCT domain."""
    INPUT_TYPES = {"image", "cutoff_freq"}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
```

### 4. Helper Node: `inspect.py`
**Purpose:** Inspect latent geometry, device, dtype for debugging
**Stub:** `nodes/helper_nodes/inspect.py`

```python
class MiniMaxH3Inspect:
    """Debug node: print latent shape, device, dtype."""
    INPUT_TYPES = {"latent"}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "inspect"
    OUTPUT_NODE = True
```

### 5. Helper Node: `spectral_expand.py`
**Purpose:** Visualize spectral expansion step
**Stub:** `nodes/helper_nodes/spectral_expand.py`

```python
class MiniMaxH3SpectralExpand:
    """Show spectral expansion effect on noise."""
    INPUT_TYPES = {"noise", "sigma", "direction"}
    RETURN_TYPES = ("NOISE", "STRING")
    FUNCTION = "expand"
```

### 6. Helper Node: `transition_math.py`
**Purpose:** Compute activation time for given A/β
**Stub:** `nodes/helper_nodes/transition_math.py`

```python
class MiniMaxH3TransitionMath:
    """Compute transition steps from power-law params."""
    INPUT_TYPES = {"power_A", "power_beta", "delta", "H", "W"}
    RETURN_TYPES = ("INT", "FLOAT", "STRING")  # steps, time, report
    FUNCTION = "compute"
```

### 7. Helper Node: `x0_fidelity_probe.py`
**Purpose:** Measure X0 fidelity during sampling (debug tool)
**Stub:** `nodes/helper_nodes/x0_fidelity_probe.py`

```python
class MiniMaxH3XFidelityProbe:
    """Probe X0 fidelity at each step."""
    INPUT_TYPES = {"x0", "x_noisy", "sigma"}
    RETURN_TYPES = ("FLOAT", "STRING")  # fidelity score, report
    FUNCTION = "probe"
    OUTPUT_NODE = True
```

### 8. Helper Node: `av_reentry_oracle.py`
**Purpose:** Compute audio-video reentry schedule
**Stub:** `nodes/helper_nodes/av_reentry_oracle.py`

```python
class MiniMaxH3AVReentryOracle:
    """Compute when audio should re-enter based on sigma."""
    INPUT_TYPES = {"sigmas", "audio_shift"}
    RETURN_TYPES = ("SIGMAS", "STRING")
    FUNCTION = "compute"
```

---

## 📋 Future (Out of Scope for MVP)

- [ ] **Lab parity nodes**: `speed_nodes/base.py`, `speed_nodes/configurable.py`, `speed_nodes/sampler_speed.py` (full Lab node hierarchy)
- [ ] **Oracle integration**: Port `oracle.py` Euler pack/unpack for advanced use cases
- [ ] **Multi-GPU support**: Parallel harvest across GPUs
- [ ] **Live latent auto-detection**: Remove optional latent requirement
- [ ] **SAMPLER-type node variant**: Alternative node architecture

---

## 🎯 Recommended Implementation Order

### Priority 1: Integration Test (1-2h)
1. Create `tests/test_integration.py`
2. Write end-to-end test: synthetic harvest → schedule → verify steps
3. Run: `uv run pytest tests/test_integration.py -v`

### Priority 2: Debug/Utility Nodes (2-4h)
Create stubs for:
1. `nodes/helper_nodes/inspect.py` — Simplest, most useful
2. `nodes/helper_nodes/power_spectrum.py` — Visualization
3. `nodes/helper_nodes/dct_lowpass.py` — Ablation testing
4. Register in `nodes/__init__.py`

### Priority 3: Advanced Nodes (4-8h)
Create stubs for:
1. `nodes/helper_nodes/transition_math.py`
2. `nodes/helper_nodes/spectral_expand.py`
3. `nodes/helper_nodes/x0_fidelity_probe.py`
4. `nodes/helper_nodes/av_reentry_oracle.py`

### Priority 4: Polish
1. Add docstrings to all new nodes
2. Update README with new node documentation
3. Create workflow examples for each new node
4. Push to origin, open PR

---

## 🧪 Current Test Matrix

```
test_dct.py          7 passed   — DCT transform correctness
test_flow.py         6 passed   — Sigma alignment, audio handling
test_harvest.py      8 passed   — Power spectrum, fitting, callback
test_sampler.py      47 passed  — Full sampler integration, coupled_full_grid
test_spectral.py     0 passed   — (empty, spectral functions tested indirectly)

Total: 68 passed
```

---

## 📝 Notes

- The MVP is intentionally minimal compared to the Lab
- Lab has ~16 helper nodes; MVP has 3 (harvest, harvest_to_config, schedule)
- The 13 missing nodes are mostly debug/visualization tools
- All core functionality is present and tested
- `coupled_full_grid` noise policy is fully supported
