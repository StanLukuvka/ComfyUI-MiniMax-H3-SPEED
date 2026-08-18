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
# NOTE: the package is named h3_speed_nodes (NOT "nodes") on purpose.
# ComfyUI's core already owns the "nodes" module (ComfyUI/nodes.py), so a
# package named "nodes" here would silently import ComfyUI's built-in node
# registry instead of ours — the sampler node still registers, but every
# helper node (SigmaHarvest, Schedule, etc.) silently disappears.
try:
    from h3_speed_nodes import (
        NODE_CLASS_MAPPINGS as _HELPER_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _HELPER_DISPLAY,
    )
    NODE_CLASS_MAPPINGS.update(_HELPER_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_HELPER_DISPLAY)
except Exception as exc:
    print(f"[MiniMaxH3SPEED] Warning: could not load helper nodes: {exc}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
