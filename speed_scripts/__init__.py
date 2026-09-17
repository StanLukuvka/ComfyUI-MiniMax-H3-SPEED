"""Core SPEED runtime support for MiniMax-H3.

Package layout:
- config.py — validated SPEED runtime configuration
- flow.py — transition-coordinate and audio-state math
- spectral.py — DCT primitives and spectral expansion
- sampler_support.py — public sampler names and run-scoped handles
- res_multistep_adapter.py — stateful RES implementation for SPEED stages
- h3_runtime.py — multi-stage SPEED pipeline
- harvest.py — native-sampler calibration math
"""
