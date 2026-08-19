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

print("MiniMax-H3 SPEED node pack: registering nodes...")

# PR3 startup banner: confirms the running code, progress-bar setting, and the
# resolved disable_pbar behavior at node-load time. If you do not see this in
# the ComfyUI console after a fresh restart, the install is stale (Python is
# still importing the old __init__.py from before the PR3 fix).
try:
    import subprocess
    _commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    _commit = "unknown (not a git checkout?)"

try:
    import comfy.utils as _comfy_utils
    _pbar_enabled = bool(_comfy_utils.PROGRESS_BAR_ENABLED)
    _sampler_disable_pbar = not _pbar_enabled
except Exception:
    _pbar_enabled = None
    _sampler_disable_pbar = None

print(f"[SPEED] git commit: {_commit}")
print(f"[SPEED] PROGRESS_BAR_ENABLED: {_pbar_enabled}")
print(f"[SPEED] sampler_node passes disable_pbar={_sampler_disable_pbar} to run_repeated_stage_calls")
print(f"[SPEED] run_repeated_stage_calls default: disable_pbar=False (bar visible by default)")
print(f"[SPEED] -> progress bar will show: {_pbar_enabled}" if _pbar_enabled is not None
      else "[SPEED] -> comfy.utils not importable at __init__ time; bar will be decided by sampler_node's runtime check")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from sampler_node import (  # noqa: E402
        NODE_CLASS_MAPPINGS as _SAMPLER_CLASS,
        NODE_DISPLAY_NAME_MAPPINGS as _SAMPLER_DISPLAY,
    )
    NODE_CLASS_MAPPINGS.update(_SAMPLER_CLASS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_SAMPLER_DISPLAY)
    print("Registered sampler_node:", sorted(_SAMPLER_CLASS))
except Exception:
    print("FAILED to import sampler_node:\n%s" % traceback.format_exc())

# All other nodes — flat root files, same pattern as sampler_node.
_NODE_MODULES = (
    "harvest_to_config_node",
)

for _name in _NODE_MODULES:
    try:
        _mod = importlib.import_module(_name)
    except Exception:
        print("FAILED to import %s:\n%s" % (_name, traceback.format_exc()))
        continue
    _mappings = getattr(_mod, "NODE_CLASS_MAPPINGS", {})
    _display = getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)
    print("Registered %-28s %s" % (_name, ", ".join(sorted(_mappings)) or "(nothing exported)"))

print("MiniMax-H3 SPEED registration complete: %d node(s)" % len(NODE_CLASS_MAPPINGS))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
