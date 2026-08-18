"""Shared helpers for MiniMax-H3 SPEED nodes."""

from __future__ import annotations

import inspect


class MockNested:
    """Lightweight mock nested tensor for helper nodes that need a return value."""

    is_nested = True

    def __init__(self, streams):
        self._streams = list(streams)

    def unbind(self):
        return self._streams


def reconstruct_nested(original, new_streams):
    """Reconstruct a NestedTensor-compatible object from new streams.

    Preserves the upstream object's class so real ComfyUI NestedTensor
    instances stay NestedTensor, and duck-typed test fixtures stay their
    own type.  Handles two constructor conventions:

    1. ``NestedTensor(list_of_tensors)`` — the real ComfyUI pattern.
    2. ``Fixture(video, audio)`` — positional streams, used by test mocks.
    """
    cls = type(original)
    try:
        return cls(list(new_streams))
    except TypeError:
        sig = inspect.signature(cls.__init__)
        n_params = sum(
            1 for p in sig.parameters.values()
            if p.name != "self"
            and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        streams = list(new_streams)
        if n_params and len(streams) >= n_params:
            return cls(*streams[:n_params])
        if n_params:
            return cls(*streams)
        # Fall back to MockNested if all else fails.
        return MockNested(streams)
