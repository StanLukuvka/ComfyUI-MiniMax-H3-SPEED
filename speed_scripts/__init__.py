"""Core SPEED runtime support.

Package layout:
- config.py — validated SPEED runtime configuration
- planning.py — stage geometry, boundary scheduling, and node config builders
- flow.py — transition-coordinate math for stage changes
- spectral.py — DCT primitives and spectral expansion
- latent_class.py — generation-local I2V keyframe lifecycle
- runtime.py — multi-stage SPEED execution
- families/ — per-model latent handling package: __init__.py registry,
  h3_video.py, flow_image.py
- harvest.py — native-sampler calibration math
"""
