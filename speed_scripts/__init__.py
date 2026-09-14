"""speed_scripts — core SPEED library for MiniMax-H3.

Package layout:
- config.py     — SpeedConfig, SCALE_PRESETS
- flow.py       — transition math (scale_ratio alignment)
- spectral.py   — DCT primitives
- sampler_support.py — public sampler names and run-scoped handles
- res_multistep_adapter.py — stateful RES implementation for SPEED stages
- h3_runtime.py — run_speed_pipeline (sampler-aware core loop)
"""

# FLOW-PRODUCED: sampler-aware SPEED runtime package.
