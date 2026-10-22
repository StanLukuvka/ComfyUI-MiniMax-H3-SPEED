"""Register the SPEED nodes with ComfyUI.

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

print("SPEED node pack: registering nodes...")

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

# Old MiniMax-H3 node IDs stay registered as deprecated subclasses. Registering
# them as replacements lets the frontend offer to swap them for the new nodes.
_REPLACED_IDS = {
    "MiniMaxH3SPEEDSampler": "SPEEDSampler",
    "MiniMaxH3SPEEDSamplerManual": "SPEEDSamplerManual",
    "MiniMaxH3HarvestToConfig": "SPEEDSigmaHarvest",
}
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _register_replacements():
    try:
        from comfy_api.latest import io
        from server import PromptServer

        manager = PromptServer.instance.node_replace_manager
    except (ImportError, AttributeError):
        return  # ComfyUI without node replacements; the deprecated IDs still load.
    for old_id, new_id in _REPLACED_IDS.items():
        if old_id not in NODE_CLASS_MAPPINGS or new_id not in NODE_CLASS_MAPPINGS:
            continue
        spec = NODE_CLASS_MAPPINGS[new_id].INPUT_TYPES()
        inputs = [name for group in ("required", "optional") for name in spec.get(group, {})]
        widgets = [
            name
            for group in ("required", "optional")
            for name, (kind, *_) in spec.get(group, {}).items()
            if isinstance(kind, list) or kind in _WIDGET_TYPES
        ]
        outputs = len(NODE_CLASS_MAPPINGS[new_id].RETURN_TYPES)
        manager.register(io.NodeReplace(
            new_node_id=new_id,
            old_node_id=old_id,
            old_widget_ids=widgets,
            input_mapping=[{"new_id": name, "old_id": name} for name in inputs],
            output_mapping=[{"new_idx": i, "old_idx": i} for i in range(outputs)],
        ))


_register_replacements()

print("SPEED registration complete: %d node(s)" % len(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
