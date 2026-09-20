"""Register the MiniMax-H3 SPEED nodes with ComfyUI.

Add this package and ``nodes/`` to ``sys.path`` before importing the node
modules by name.
"""

import importlib
import os
import sys
import traceback

# Node modules are imported by name, so both folders must be on sys.path.
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_NODES_DIR = os.path.join(_NODE_DIR, "nodes")
for _p in (_NODE_DIR, _NODES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print("MiniMax-H3 SPEED node pack: registering nodes...")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _register(_mod, _name):
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    _display = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)
    print("Registered %-28s %s" % (_name, ", ".join(sorted(_mappings)) or "(nothing exported)"))


# The pack exposes exactly these three node modules.
_NODE_MODULES = (
    "sampler_node",
    "sampler_node_manual",
    "sampler_sigma_harvest_node",
)

for _name in _NODE_MODULES:
    try:
        _mod = importlib.import_module(_name)
    except Exception:
        print("FAILED to import %s:\n%s" % (_name, traceback.format_exc()))
        continue
    _register(_mod, _name)

print("MiniMax-H3 SPEED registration complete: %d node(s)" % len(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
