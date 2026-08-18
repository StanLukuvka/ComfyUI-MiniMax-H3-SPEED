"""Logging for the H3-SPEED node pack.

ComfyUI swallows custom-node import errors silently — a broken node pack
just fails to register and the only user symptom is "custom node not
installed". To make failures visible, we emit loud, greppable banners:
  - fall back to ComfyUI's logger where present
  - otherwise fall back to stderr with a [MINIMAX-H3-SPEED] prefix.

Each configured logger owns its handler and never propagates, so child
loggers (e.g. ``MinimaxH3.SPEED.Harvest``) don't get swallowed or
double-printed by the root logger.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "MinimaxH3.SPEED"
_FORMAT = "[MINIMAX-H3-SPEED] %(asctime)s %(levelname)s %(name)s: %(message)s"
_LEVEL = logging.INFO


def _configure(logger: logging.Logger) -> logging.Logger:
    logger.setLevel(_LEVEL)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    logger._h3_speed_configured = True  # type: ignore[attr-defined]
    return logger


def get_logger(child: str | None = None) -> logging.Logger:
    """Return a configured pack logger (or a named child of it).

    ``get_logger("Harvest")`` returns ``MinimaxH3.SPEED.Harvest`` with its own
    handler, so modules inside ``minimax_h3_speed/`` can log without importing
    anything from the repo root.
    """
    name = LOGGER_NAME if child is None else f"{LOGGER_NAME}.{child}"
    logger = logging.getLogger(name)
    if getattr(logger, "_h3_speed_configured", False):
        return logger
    return _configure(logger)


def banner(text: str) -> None:
    """Unmissable banner for registration/failure events."""
    get_logger().info("=" * 72)
    get_logger().info(text)
    get_logger().info("=" * 72)
