# Sigma Variant Plan — ComfyUI-MiniMax-H3-SPEED

**Branch:** `feature/sigma-variant` (off `main`)
**Goal:** Port the sigma-harvest + delta-custom calibration pipeline from Lab to the MVP sampler.

---

## Context

The MVP currently only supports `transition_mode="explicit"` with hardcoded `DEFAULT_TRANSITION_STEPS`. The Lab has a full calibration pipeline:

1. **SigmaHarvest** — runs one Euler pass, captures residual power spectrum at each sigma
2. **HarvestToConfig** — parses the JSON, fits P=A·ω^(-β), emits a readable report
3. **Schedule node** — pre-computes delta-optimal transition steps from fitted A/β
4. **delta_custom mode** — runtime uses fitted A/β to compute steps on-the-fly

The MVP is missing all of the above. This plan ports them.

---

## Files to Create

### 1. `minimax_h3_speed/harvest.py` — Radial power spectrum + fitting

```python
def radial_power_spectrum(video: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Mean 2D-DCT power of [B,C,T,H,W] binned radially. Returns (freqs, profile)."""

def fit_power_law(freqs, profile, omega_min=0.5) -> dict:
    """Fit P = A * ω^(-β) on log-log. Returns {A, beta, r_squared, n_bins}."""

def _fit_health(fit: dict) -> str:
    """good/fair/weak/suspect based on beta > 0 and r² threshold."""

@dataclass
class HarvestCallback:
    """Captures residual (x - x0) power per sigma during guider.sample()."""
    sigmas: torch.Tensor
    every: int = 1
    profiles: list = field(default_factory=list)
    
    def __call__(self, step, x0, x, total_steps): ...
    def finalize(self, omega_min, latent_h, latent_w, delta, fit_mode) -> dict:
        """Returns payload with overall_fit_A, overall_fit_beta, recommended_config."""

def recommend_configs(A, beta, sigmas, latent_h, latent_w, delta=0.01) -> dict:
    """For each preset, compute delta-optimal transition_steps using fitted A/β."""
```

**Dependencies:** `spectral.dct2`, `h3_runtime.power_spectrum`, `h3_runtime.activation_time`, `config.SCALE_PRESETS`.

**Tests:** `tests/test_harvest.py` — 8 tests covering:
- `radial_power_spectrum` returns correct shape
- `fit_power_law` recovers known A/β
- `HarvestCallback` captures at correct intervals
- `finalize` with `fit_mode="first"` / `"per_sigma"` / `"pooled"`
- `recommend_configs` produces valid step indices

---

### 2. `nodes/helper_nodes/sigma_harvest.py` — ComfyUI node

```python
class MiniMaxH3SigmaHarvest:
    INPUT_TYPES: noise, guider, sigmas, latent_image, capture_every, fit_mode, omega_min, delta
    RETURN_TYPES: ("STRING",)
    FUNCTION: "run"
    CATEGORY: "sampling/minimax_h3_speed/sigma_harvest"
    OUTPUT_NODE: True
```

**Behavior:**
1. Validate H3 nested latent via `_ensure_h3`
2. Zero the latent values (keep shape/device/dtype) — measured power must be from clean noise, not content
3. Run one Euler pass with `HarvestCallback`
4. Call `finalize()` with user parameters
5. Return JSON string → wire to `SaveText`

---

### 3. `nodes/helper_nodes/harvest_to_config.py` — ComfyUI node

```python
class MiniMaxH3HarvestToConfig:
    INPUT_TYPES: harvest_json (STRING, default "{}")
    RETURN_TYPES: ("STRING",)
    FUNCTION: "parse"
    CATEGORY: "sampling/minimax_h3_speed/sigma_harvest"
    OUTPUT_NODE: True
```

**Behavior:**
1. Parse JSON, validate `overall_fit_A` and `overall_fit_beta`
2. Emit plain-text report: A, β, r², fit_health, recommended steps per preset
3. Warn on suspect/weak fits

---

### 4. `nodes/helper_nodes/schedule.py` — ComfyUI Schedule node

```python
class MiniMaxH3SPEEDSchedule:
    INPUT_TYPES: sigmas, preset, transition_mode (manual_step/manual_sigma/delta_custom),
                 manual_sigma (FLOAT), delta (FLOAT), power_A (FLOAT), power_beta (FLOAT),
                 full_latent_h (INT), full_latent_w (INT)
    RETURN_TYPES: ("H3_SPEED_CONFIG", "STRING")
    FUNCTION: "plan"
    CATEGORY: "sampling/minimax_h3_speed/schedule"
```

**Behavior:**
- Computes transition steps from sigmas + preset + mode
- `manual_step`: use DEFAULT_TRANSITION_STEPS as-is
- `manual_sigma`: find first sigma ≤ manual_sigma for each transition
- `delta_custom`: compute activation time from fitted A/β, find first step below
- Returns `SpeedConfig` + human-readable report string

---

### 5. Update `sampler_node.py` — Add optional latent + delta_custom

**Changes:**
1. Add `optional={"latent": ("LATENT",)}` to `INPUT_TYPES`
2. When `transition_mode="delta_custom"`, call `resolve_transition_steps(config, sigmas, H_full, W_full)` using live latent dims if provided
3. Expose `delta`, `power_A`, `power_beta`, `transition_seed_offset` widgets when `delta_custom` is selected
4. Guard: reject `delta_custom` if latent dims unavailable (or fall back to config defaults)

---

### 6. Update `h3_runtime.py` — delta_custom support

The MVP's `resolve_transition_steps` already handles `delta_custom`:
```python
if config.transition_mode == "delta_custom":
    # Uses config.power_A, config.power_beta, config.delta
    # Needs H_full, W_full — use live latent dims if passed, else config defaults
```

**One change needed:** Accept `H_full, W_full` as optional kwargs:
```python
def resolve_transition_steps(config, sigmas, H_full=None, W_full=None):
    if config.transition_mode != "delta_custom":
        return list(config.transition_steps)
    if H_full is None or W_full is None:
        H_full, W_full = config.full_latent_h, config.full_latent_w
    # ... rest unchanged
```

---

### 7. Update `config.py` — Expose all controls

**Current state:** Only `preset` and `transition_mode` exposed.
**Target state:** All SpeedConfig fields available as node widgets when in advanced mode.

```python
# In sampler_node.py INPUT_TYPES:
"delta": ("FLOAT", {"default": 0.01, "min": 1e-6, "max": 0.999999, "step": 0.001}),
"power_A": ("FLOAT", {"default": 219.48, "min": 1e-6, "max": 1e6}),
"power_beta": ("FLOAT", {"default": 2.42, "min": 1e-6, "max": 10.0}),
"transition_seed_offset": ("INT", {"default": 10000, "min": 1, "max": 0x7FFFFFFF}),
```

**Conditional visibility:** These show only when `transition_mode == "delta_custom"`.
(ComfyUI doesn't have conditional widgets, so show them always but document their use.)

---

### 8. Update `__init__.py` — Register new nodes

```python
from .helper_nodes.sigma_harvest import MiniMaxH3SigmaHarvest
from .helper_nodes.harvest_to_config import MiniMaxH3HarvestToConfig
from .helper_nodes.schedule import MiniMaxH3SPEEDSchedule

NODE_CLASS_MAPPINGS.update({
    "MiniMaxH3SigmaHarvest": MiniMaxH3SigmaHarvest,
    "MiniMaxH3HarvestToConfig": MiniMaxH3HarvestToConfig,
    "MiniMaxH3SPEEDSchedule": MiniMaxH3SPEEDSchedule,
})
```

---

## Files to Create (Workflows)

### 9. `workflows/sigma_harvest.json`
Basic harvest workflow:
- UNETLoader → BasicScheduler → RandomNoise → BasicGuider → MiniMaxH3ImageToVideo
- Output → MiniMaxH3SigmaHarvest → SaveText

### 10. `workflows/sigma_harvest_calibrated.json`
Full pipeline:
- Same as above + MiniMaxH3HarvestToConfig
- Shows calibrated A/β and recommended steps per preset

---

## Testing Strategy

**Unit tests** (`tests/test_harvest.py`):
- DCT power spectrum correctness
- Power-law fitting accuracy
- Callback captures at right intervals
- Finalize modes produce correct output
- Recommend configs produce valid step indices

**Integration tests** (`tests/test_sampler_delta_custom.py`):
- Sampler with `delta_custom` mode runs without crashing
- Transition steps computed match expected values
- Parity with Lab when given same A/β

**Workflow test:**
- Load `sigma_harvest.json` in ComfyUI, verify harvest JSON is valid
- Load `sigma_harvest_calibrated.json`, verify report is readable

---

## Implementation Order

1. **harvest.py** — core math, no ComfyUI deps
2. **test_harvest.py** — verify math independently
3. **sigma_harvest.py** — ComfyUI node
4. **harvest_to_config.py** — ComfyUI node
5. **schedule.py** — ComfyUI node
6. **Update sampler_node.py** — add latent input + delta_custom widgets
7. **Update h3_runtime.py** — accept H_full/W_full kwargs
8. **Update __init__.py** — register new nodes
9. **Create workflows** — harvest + calibrated
10. **Final integration tests** — end-to-end verification

---

## Not In Scope (Future)

- `coupled_full_grid` noise policy — supported in Lab but architecture requires changes
- SAMPLER-type node variant — needs ComfyUI-level changes
- Live latent geometry auto-detection without optional input
- Multi-GPU parallel harvest

---

## Success Criteria

- [ ] 22 existing tests still pass
- [ ] 8+ new harvest tests pass
- [ ] Sampler node accepts `delta_custom` mode without crashing
- [ ] Harvest node produces valid JSON
- [ ] Schedule node produces valid SpeedConfig
- [ ] End-to-end workflow: harvest → read report → paste into sampler
- [ ] README updated with new nodes and workflow
