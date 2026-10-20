"""Core SPEED runtime support for MiniMax-H3.

Package layout:
- config.py — validated SPEED runtime configuration
- planning.py — stage geometry, boundary scheduling, and node config builders
- flow.py — transition-coordinate and audio-state math
- spectral.py — DCT primitives and spectral expansion
- latent_class.py — generation-local I2V keyframe lifecycle
- runtime.py — multi-stage SPEED execution
- families.py — MiniMax-H3 latent helpers
- harvest.py — native-sampler calibration math
"""
