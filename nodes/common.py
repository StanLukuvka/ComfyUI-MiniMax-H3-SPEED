"""Shared helpers for MiniMax-H3 SPEED nodes."""

from __future__ import annotations


class MockNested:
    """Lightweight mock nested tensor for helper nodes that need a return value."""

    is_nested = True

    def __init__(self, streams):
        self._streams = list(streams)

    def unbind(self):
        return self._streams
