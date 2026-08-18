"""ComfyUI custom node package for MiniMax H3 SPEED.

ComfyUI loads this directory as a package via ``importlib.util.spec_from_file_location``,
which does NOT put this directory on ``sys.path``. We add it explicitly so the
node-class modules are importable by name — the exact same pattern ComfyUI
itself uses for its built-in nodes. Every node is a flat file here, imported
directly (same as sampler_node.py which always worked).
"""

import importlib
import logging
import os
import sys
import traceback

# Ensure our directory + repo root are importable so `from common import ...`
# and `from minimax_h3_speed... import ...` resolve when ComfyUI imports this
# module (it loads sibling modules by flat name, NOT by package path).
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.abspath(_NODE_DIR))
for _p in (_NODE_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from h3_logging import get_logger, banner  # noqa: E402

log = get_logger()

banner("MiniMax-H3 SPEED node pack: registering nodes...")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from sampler_node import (  # noqa: E402
        NODE_CLASS_MAPPINGS as _SAMPLER_CLASS,
        NODE_DISPLAY_NAME_MAPPINGS as _SAMPLER_DISPLAY,
    )
    NODE_CLASS_MAPPINGS.update(_SAMPLER_CLASS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_SAMPLER_DISPLAY)
    log.info("Registered sampler_node: %s", sorted(_SAMPLER_CLASS))
except Exception:
    log.error("FAILED to import sampler_node:\n%s", traceback.format_exc())

# All other nodes — flat root files, same pattern as sampler_node.
_NODE_MODULES = (
    "sigma_harvest_node",
    "harvest_to_config_node",
    "schedule_node",
    "inspect_node",
    "power_spectrum_node",
    "dct_lowpass_node",
    "transition_math_node",
    "spectral_expand_node",
    "x0_fidelity_probe_node",
    "av_reentry_oracle_node",
)

for _name in _NODE_MODULES:
    try:
        _mod = importlib.import_module(_name)
    except Exception:
        log.error("FAILED to import %s:\n%s", _name, traceback.format_exc())
        continue
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    _display = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)
    log.info("Registered %-28s %s", _name, ", ".join(sorted(_mappings)) or "(nothing exported)")

banner(f"MiniMax-H3 SPEED registration complete: {len(NODE_CLASS_MAPPINGS)} node(s)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
