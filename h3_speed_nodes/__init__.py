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
try:
    from h3_speed_nodes.helper_nodes import (
        sigma_harvest,
        harvest_to_config,
        schedule,
        inspect,
        power_spectrum,
        dct_lowpass,
        transition_math,
        spectral_expand,
        x0_fidelity_probe,
        av_reentry_oracle,
    )
except Exception as exc:
    print(f"[MiniMaxH3SPEED] Warning: could not load helper nodes: {exc}")
else:
    NODE_CLASS_MAPPINGS.update({
        "MiniMaxH3SigmaHarvest": sigma_harvest.MiniMaxH3SigmaHarvest,
        "MiniMaxH3HarvestToConfig": harvest_to_config.MiniMaxH3HarvestToConfig,
        "MiniMaxH3SPEEDSchedule": schedule.MiniMaxH3SPEEDSchedule,
        "MiniMaxH3Inspect": inspect.MiniMaxH3Inspect,
        "MiniMaxH3PowerSpectrum": power_spectrum.MiniMaxH3PowerSpectrum,
        "MiniMaxH3DCTLowpass": dct_lowpass.MiniMaxH3DCTLowpass,
        "MiniMaxH3TransitionMath": transition_math.MiniMaxH3TransitionMath,
        "MiniMaxH3SpectralExpand": spectral_expand.MiniMaxH3SpectralExpand,
        "MiniMaxH3XFidelityProbe": x0_fidelity_probe.MiniMaxH3XFidelityProbe,
        "MiniMaxH3AVReentryOracle": av_reentry_oracle.MiniMaxH3AVReentryOracle,
    })
    for module in (sigma_harvest, harvest_to_config, schedule, inspect,
                   power_spectrum, dct_lowpass, transition_math,
                   spectral_expand, x0_fidelity_probe, av_reentry_oracle):
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
