"""Global logging configuration for agent_harness."""
from __future__ import annotations

import logging

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the agent_harness framework.

    Sets up a root logger with a formatter that includes timestamp, level,
    module name, line number, and message.  Third-party libraries (openai,
    httpx, urllib3) are silenced to WARNING to reduce noise.

    Can be called multiple times to change the level.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s:%(lineno)d %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        root = logging.getLogger()
        root.addHandler(handler)
        root.setLevel(numeric_level)

        # Suppress noisy third-party loggers
        for name in ("openai", "httpx", "urllib3", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

        _configured = True
    else:
        logging.getLogger().setLevel(numeric_level)
