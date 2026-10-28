"""Core SPEED runtime support.

Package layout:
- config.py — validated SPEED runtime configuration
- planning.py — stage geometry, boundary scheduling, and node config builders
- flow.py — transition-coordinate math for stage changes
- spectral.py — DCT primitives and spectral expansion
- latent_class.py — generation-local I2V keyframe lifecycle
- runtime.py — multi-stage SPEED execution
- families/ — per-model latent handling (registry, h3_video, flow_image)
- harvest.py — native-sampler calibration math
"""
