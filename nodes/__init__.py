"""ComfyUI custom node package for MiniMax H3 SPEED.

ComfyUI loads this directory as a package via ``importlib.util.spec_from_file_location``,
which does NOT put this directory on ``sys.path``. We add it explicitly so the
node-class module (``sampler_node.py``) is importable by name.
"""

import os
import sys

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from sampler_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Import helper nodes to register them in the global namespace.
# These are loaded by ComfyUI's node discovery but don't need explicit
# registration here since each module defines its own NODE_CLASS_MAPPINGS.
try:
    from nodes.helper_nodes import sigma_harvest, harvest_to_config, schedule
except Exception as exc:  # helper nodes need comfy available at runtime
    print(f"[MiniMaxH3SPEED] Warning: could not load helper nodes: {exc}")

NODE_CLASS_MAPPINGS.update({
    "MiniMaxH3SigmaHarvest": sigma_harvest.MiniMaxH3SigmaHarvest,
    "MiniMaxH3HarvestToConfig": harvest_to_config.MiniMaxH3HarvestToConfig,
    "MiniMaxH3SPEEDSchedule": schedule.MiniMaxH3SPEEDSchedule,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "MiniMaxH3SigmaHarvest": sigma_harvest.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3SigmaHarvest"],
    "MiniMaxH3HarvestToConfig": harvest_to_config.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3HarvestToConfig"],
    "MiniMaxH3SPEEDSchedule": schedule.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3SPEEDSchedule"],
})

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
