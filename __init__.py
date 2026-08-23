"""ComfyUI custom node package for MiniMax H3 SPEED.

ComfyUI loads this directory as a package via ``importlib.util.spec_from_file_location``,
which does NOT put this directory on ``sys.path``. We add it explicitly so the
node-class modules are importable by name — the exact same pattern ComfyUI
itself uses for its built-in nodes.

Layout (after the reorganise-folder-structure commit):

- ``nodes/`` — one file per ComfyUI node (flat, imported by name).
- ``speed_scripts/`` — the core SPEED library package (config, flow, spectral,
  harvest, h3_runtime). Installed editable into this repo's venv so the
  ``from speed_scripts... import ...`` sites inside the node files resolve.
"""

import importlib
import logging
import os
import sys
import traceback

# Ensure our directory + repo root are importable so `from speed_scripts...`
# and flat node-module imports resolve when ComfyUI imports this module (it
# loads sibling modules by flat name, NOT by package path).
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_NODES_DIR = os.path.join(_NODE_DIR, "nodes")
for _p in (_NODE_DIR, _NODES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print("MiniMax-H3 SPEED node pack: registering nodes...")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _register(_name, _mod, _dbg):
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    _display = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)
    print("Registered %-28s %s" % (_dbg, ", ".join(sorted(_mappings)) or "(nothing exported)"))


# All nodes — flat files under nodes/.
_NODE_MODULES = (
    "sampler_node",
    "sampler_sigma_harvest_node",
)

for _name in _NODE_MODULES:
    try:
        _mod = importlib.import_module(_name)
    except Exception:
        print("FAILED to import %s:\n%s" % (_name, traceback.format_exc()))
        continue
    _register(_name, _mod, _name)

print("MiniMax-H3 SPEED registration complete: %d node(s)" % len(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
