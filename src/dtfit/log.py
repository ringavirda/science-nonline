"""Library logging.

dtfit logs through the standard :mod:`logging` module under the ``"dtfit"``
logger. Following library best practice the logger has only a ``NullHandler``
attached, so importing dtfit never configures the root logger or prints
anything by itself. Applications opt in to output; ``enable_logging`` is a
convenience for notebooks and scripts.
"""

import logging

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
