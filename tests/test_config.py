"""Tests for environment-driven configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from resume_classifier.config import (
    PROJECT_ROOT,
    Environment,
    Settings,
    get_settings,
    unknown_dotenv_keys,
)


def test_defaults_are_safe() -> None:
    """A process started with no environment at all must be safe by default."""
    settings = Settings(_env_file=None)

    assert settings.app_name == "resume-classifier"
    assert settings.environment is Environment.DEVELOPMENT
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.database_enabled is False
    assert settings.store_resume_text is False


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_ENVIRONMENT", "production")
    monkeypatch.setenv("RC_LOG_LEVEL", "WARNING")

    settings = Settings(_env_file=None)

    assert settings.environment is Environment.PRODUCTION
    assert settings.is_production is True
    assert settings.log_level == "WARNING"


def test_debug_is_rejected_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """Debug mode in production is an RCE vector and must fail at startup."""
    monkeypatch.setenv("RC_ENVIRONMENT", "production")
    monkeypatch.setenv("RC_DEBUG", "true")

    with pytest.raises(ValidationError, match="RC_DEBUG must be false"):
        Settings(_env_file=None)


def test_debug_is_allowed_outside_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_ENVIRONMENT", "development")
    monkeypatch.setenv("RC_DEBUG", "true")

    assert Settings(_env_file=None).debug is True


def test_unknown_environment_variable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo in an RC_* variable must fail loudly, not be silently ignored."""
    monkeypatch.setenv("RC_LOG_LEVELL", "DEBUG")

    with pytest.raises(ValidationError, match="unknown environment variable"):
        Settings(_env_file=None)


def test_invalid_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RC_ENVIRONMENT", "staging")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_split_proportions_must_partition_the_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RC_SPLIT_TRAIN", "0.5")
    monkeypatch.setenv("RC_SPLIT_VALIDATION", "0.2")
    monkeypatch.setenv("RC_SPLIT_TEST", "0.2")

    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        Settings(_env_file=None)


def test_derived_data_paths() -> None:
    settings = Settings(_env_file=None)

    assert settings.raw_data_dir == settings.data_dir / "raw"
    assert settings.processed_data_dir == settings.data_dir / "processed"
    assert settings.taxonomy_dir == settings.data_dir / "taxonomy"
    assert settings.dataset_dir == settings.processed_data_dir / settings.dataset_version
    assert settings.data_dir.parent == settings.project_root
    assert settings.models_dir.parent == settings.project_root


def test_database_is_enabled_only_by_a_url() -> None:
    assert Settings(_env_file=None).database_enabled is False
    assert Settings(_env_file=None, database_url="postgresql://x/y").database_enabled is True


def test_settings_are_immutable() -> None:
    """Frozen settings keep runtime behaviour deterministic."""
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError):
        settings.debug = True  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


# ===========================================================================
# The shared .env contract
#
# A project's .env is read by BOTH the application and Docker Compose. Compose
# needs its own substitution variables in that same file, so foreign keys must
# be tolerated -- while the RC_ namespace, which belongs to the application,
# stays strict.
# ===========================================================================
def test_foreign_keys_in_a_shared_dotenv_are_tolerated(tmp_path: Path) -> None:
    """Rejecting POSTGRES_* would make `cp .env.example .env` break the app."""
    env_file = tmp_path / "shared.env"
    env_file.write_text(
        "POSTGRES_DB=resume_db\n"
        "POSTGRES_PASSWORD=secret\n"
        "COMPOSE_BIND_HOST=127.0.0.1\n"
        "RC_LOG_LEVEL=WARNING\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.log_level == "WARNING"


def test_unknown_rc_key_in_a_dotenv_is_detected(tmp_path: Path) -> None:
    """The RC_ namespace belongs to the application and stays strict."""
    env_file = tmp_path / "typo.env"
    env_file.write_text("RC_LOG_LEVELL=DEBUG\nPOSTGRES_DB=x\n", encoding="utf-8")

    assert unknown_dotenv_keys(env_file) == ["RC_LOG_LEVELL"]


def test_valid_dotenv_reports_no_stray_keys(tmp_path: Path) -> None:
    env_file = tmp_path / "good.env"
    env_file.write_text(
        "# a comment\nRC_LOG_LEVEL=INFO\nPOSTGRES_USER=resume\n\n", encoding="utf-8"
    )

    assert unknown_dotenv_keys(env_file) == []


def test_missing_dotenv_is_not_an_error(tmp_path: Path) -> None:
    assert unknown_dotenv_keys(tmp_path / "absent.env") == []


def test_shipped_env_example_is_loadable() -> None:
    """`cp .env.example .env` must produce a working application.

    This is the exact path a new user follows, and it regressed once already:
    Compose variables had been written into the RC_ namespace, so copying the
    template made the application refuse to start.
    """
    example = PROJECT_ROOT / ".env.example"

    assert unknown_dotenv_keys(example) == []
    assert Settings(_env_file=example).app_name == "resume-classifier"
