"""Library logging.

dtfit logs through the standard :mod:`logging` module under the ``"dtfit"``
logger. Following library best practice the logger has only a ``NullHandler``
attached, so importing dtfit never configures the root logger or prints
anything by itself. Applications opt in to output; ``enable_logging`` is a
convenience for notebooks and scripts.
"""

import logging
from typing import Any

logger = logging.getLogger("dtfit")
logger.addHandler(logging.NullHandler())


def enable_logging(
    level: int = logging.INFO,
    fmt: str = "%(name)s %(levelname)s: %(message)s",
) -> logging.Logger:
    """Attach a stream handler to the dtfit logger and set its level.

    Args:
        level: Logging level (e.g. ``logging.INFO`` or ``logging.DEBUG``).
        fmt: Format string for the stream handler.

    Returns:
        The configured ``"dtfit"`` logger.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def echo(
    message: str,
    value: list[Any] | Any | None = None,
    *,
    level: int = logging.DEBUG,
) -> None:
    """Emit a fitting-detail message through the dtfit logger.

    Logged at DEBUG by default and silent unless the application configures
    logging (see :func:`dtfit.enable_logging` -- use
    ``enable_logging(logging.DEBUG)`` to see fitting detail).

    Args:
        message: Message to log before any values.
        value: Value(s) to log after the message.
        level: Logging level (default ``logging.DEBUG``).
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, message)
    if value is not None:
        values = value if isinstance(value, list) else [value]
        for val in values:
            logger.log(level, "%s", val)
