"""ComfyUI custom node package for MiniMax H3 SPEED.

ComfyUI loads this directory as a package via ``importlib.util.spec_from_file_location``,
which does NOT put this directory on ``sys.path``. We add it explicitly so the
node-class modules are importable by name — the exact same pattern ComfyUI
itself uses for its built-in nodes. Every node is a flat file here, imported
directly (same as sampler_node.py which always worked).
"""

import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from sampler_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

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

import importlib
for _name in _NODE_MODULES:
    _mod = importlib.import_module(_name)
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    _display = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
