# TODO: Function-Prefix Naming Convention for Short Locals

**Status:** Not started. Parked for later.
**Date:** 2026-08-20

## Rule

Single-letter and 2-char local variables get a prefix derived from the enclosing
function name: **first letter of every word in the function name + `_`**.

Exceptions: `self`, `cls`, private names (`_*`), and names that are already
descriptive multi-word locals (`coarse_video`, `transition_steps`, etc.) are
left as-is. Loop variables in one-liner comprehensions are also fine.

## Rename Map (35 renames across 6 files)

### harvest_to_config_node.py — function `harvest` (prefix `h_`)
| Old | New    | Context |
|-----|--------|---------|
| A   | h_A    | local |
| f   | h_f    | local |
| i   | h_i    | local (loop var in sigma list comprehension) |
| r2  | h_r2   | local |
| s   | h_s    | local |

### harvest_to_config_node.py — function `extract_video_stream` (prefix `evs_`)
| Old | New    | Context |
|-----|--------|---------|
| t   | evs_t  | param |

### minimax_h3_speed/config.py — function `__post_init__` (prefix `pi_`)
| Old | New    | Context |
|-----|--------|---------|
| s   | pi_s   | local (scale iter) |

### minimax_h3_speed/flow.py — function `aligned_sigma` (prefix `as_`)
| Old | New    | Context |
|-----|--------|---------|
| q   | as_q   | local (sigma) |

### minimax_h3_speed/h3_runtime.py
| Old | New      | Function                | Context |
|-----|----------|-------------------------|---------|
| A   | paf_A    | power_at_frequency      | param |
| i   | ffsb_i   | _find_first_step_below  | local |
| n   | ffsb_n   | _find_first_step_below  | local |
| p   | rts_p    | resolve_transition_steps| local |
| q   | rsp_q    | run_speed_pipeline      | local |
| s   | ffsb_s   | _find_first_step_below  | local |
| ts  | rsp_ts   | run_speed_pipeline      | local (transition_steps short) |
| v   | rss_v    | resolve_sigma_shifts    | local |
| x   | c_x      | callback                | param |

### minimax_h3_speed/harvest.py
| Old | New      | Function                  | Context |
|-----|----------|---------------------------|---------|
| A   | rts_A    | recommend_transition_steps| param |
| H   | rdp_H    | radial_dct_power          | local |
| W   | rdp_W    | radial_dct_power          | local |
| a   | cfq_a    | classify_fit_quality     | local |
| cx  | rdp_cx   | radial_dct_power          | local |
| cy  | rdp_cy   | radial_dct_power          | local |
| i   | rts_i    | recommend_transition_steps| local |
| j   | rts_j    | recommend_transition_steps| local |
| p   | rts_p    | recommend_transition_steps| local |
| r2  | fpl_r2   | fit_power_law            | local |
| s   | rts_s    | recommend_transition_steps| local |
| x   | fpl_x    | fit_power_law            | local |
| xx  | rdp_xx   | radial_dct_power          | local |
| y   | fpl_y    | fit_power_law            | local |
| yy  | rdp_yy   | radial_dct_power          | local |

### minimax_h3_speed/spectral.py
| Old | New      | Function        | Context |
|-----|----------|-----------------|---------|
| H   | lf_H     | lowpass_filter  | local |
| T   | dt_T     | dct_temporal    | local |
| W   | lf_W     | lowpass_filter  | local |

### sampler_node.py
No renames — all locals are already descriptive.

## Notes

- `h3_runtime.py` has TWO functions with local `p` — `rsp_p` already exists
  (so resolve_transition_steps's `p` → `rts_p`, no collision).
- `rts_p` appears in both `resolve_transition_steps` functions in
  `h3_runtime.py` and `harvest.py` — these are different files, no collision.
- `c_x` in `callback` is the ComfyUI sampler callback signature — renaming it
  is fine because the callback is registered locally, not called externally.
- Lab comparison tests in `test_dct.py` (test_spectral_dct2, etc.) are
  deselected — they require the `speed_lab` sibling repo which isn't present.
- 42 tests pass after the previous rename commit; this would be a follow-up.
