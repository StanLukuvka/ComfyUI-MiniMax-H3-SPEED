# NAMES.md — Function-Prefix Naming Convention for Short Locals

**Status:** Superseded — the reorg (`ec8cdda`) moved to `speed_scripts/` +
`nodes/` and the convention was never applied. Keep this file only as a
historical note; do NOT apply the renames to the current codebase.

## Original rule (parked 2026-08-20)

Single-letter and 2-char local variables get a prefix derived from the
enclosing function name: **first letter of every word in the function name +
`_`**.

Exceptions: `self`, `cls`, private names (`_*`), descriptive multi-word
locals (`coarse_video`, `transition_steps`), and loop variables in one-liner
comprehensions.

## Why it's superseded

- The files it targeted (`harvest_to_config_node.py`, `minimax_h3_speed/`)
  were deleted/renamed in the reorg.
- House style (per AGENTS.md) is simple/ASCII, avoid one-use
  constants/helpers, comment awkward flow, defer optimization. The prefix
  convention adds noise without a clear win; the current codebase keeps
  short locals readable in context.
- If a future refactor wants it, re-derive the rename map against the
  current `speed_scripts/` + `nodes/` layout — the old map is stale.
