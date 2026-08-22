"""pytest bootstrap for the reorganised layout.

The library package moved from ``minimax_h3_speed/`` to ``speed_scripts/`` and
the node modules moved from the repo root into ``nodes/``. Put the repo root
and the ``nodes`` directory on ``sys.path`` before collection so both the
``speed_scripts.*`` package imports and the flat node-module imports
(``sampler_node``, ``sampler_sigma_harvest_node``) resolve.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../minimax_speed
NODES_DIR = REPO_ROOT / "nodes"

for _p in (str(REPO_ROOT), str(NODES_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
