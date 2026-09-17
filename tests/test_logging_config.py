"""Tests for logging configuration."""

from __future__ import annotations

import logging

from resume_classifier.config import Settings
from resume_classifier.logging_config import build_logging_config, configure_logging


def test_config_reflects_requested_log_level() -> None:
    config = build_logging_config(Settings(_env_file=None, log_level="WARNING"))

    assert config["version"] == 1
    assert config["root"]["level"] == "WARNING"
    assert config["loggers"]["resume_classifier"]["level"] == "WARNING"


def test_config_does_not_disable_existing_loggers() -> None:
    """Disabling existing loggers silently breaks third-party library logging."""
    config = build_logging_config(Settings(_env_file=None))

    assert config["disable_existing_loggers"] is False


def test_logs_are_written_to_stdout() -> None:
    """stdout is the correct sink for a containerised process."""
    config = build_logging_config(Settings(_env_file=None))

    assert config["handlers"]["console"]["stream"] == "ext://sys.stdout"


def test_configure_logging_applies_level() -> None:
    configure_logging(Settings(_env_file=None, log_level="DEBUG"))

    assert logging.getLogger("resume_classifier").level == logging.DEBUG


def test_configure_logging_is_idempotent() -> None:
    """A second call is a no-op unless explicitly forced."""
    configure_logging(Settings(_env_file=None, log_level="INFO"))
    logger = logging.getLogger("resume_classifier")
    assert logger.level == logging.INFO

    configure_logging(Settings(_env_file=None, log_level="DEBUG"))
    assert logger.level == logging.INFO

    configure_logging(Settings(_env_file=None, log_level="DEBUG"), force=True)
    assert logger.level == logging.DEBUG
