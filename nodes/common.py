"""Shared plumbing for the MiniMax H3 SPEED sampler nodes."""

from __future__ import annotations

import comfy.nested_tensor


def _nested(video, audio):
    return comfy.nested_tensor.NestedTensor([video, audio])
