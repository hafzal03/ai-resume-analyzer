"""Logging configuration.

Logging is configured once per process from :class:`~resume_classifier.config.Settings`
and writes to stdout, which is the correct sink for a containerised application.

Privacy rule for every module in this project: log decisions, identifiers, and
diagnostics -- never resume content. Uploaded documents contain names, addresses,
phone numbers, and employment history, none of which belong in a log stream.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

from resume_classifier.config import Settings, get_settings

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

_configured = False


def build_logging_config(settings: Settings) -> dict[str, Any]:
    """Build a :func:`logging.config.dictConfig` dictionary from ``settings``.

    Kept separate from :func:`configure_logging` so the resulting structure can
    be asserted in tests without mutating global logging state.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": DATE_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "resume_classifier": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Apply logging configuration, at most once per process.

    Args:
        settings: Settings to configure from. Defaults to :func:`get_settings`.
        force: Re-apply configuration even if it has already been applied.
    """
    global _configured  # module-level idempotency flag
    if _configured and not force:
        return
    resolved = get_settings() if settings is None else settings
    logging.config.dictConfig(build_logging_config(resolved))
    _configured = True


def reset_logging() -> None:
    """Clear the 'already configured' flag. Intended for use by tests."""
    global _configured  # module-level idempotency flag
    _configured = False
