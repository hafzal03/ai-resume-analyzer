"""Application configuration.

All runtime configuration arrives through environment variables prefixed with
``RC_`` (optionally via a local ``.env`` file). This module is the single place
the application learns anything about its environment: no host, path, credential
or flag is hard-coded anywhere else.

Unknown ``RC_*`` variables are rejected so that a typo fails loudly at startup
rather than silently falling back to a default.

Foreign keys are tolerated. A project's ``.env`` is shared with Docker Compose,
which needs its own substitution variables (``POSTGRES_PASSWORD`` and friends)
in that same file. Those are ignored rather than fatal -- but anything in the
``RC_`` namespace must be a real setting, because that namespace is ours.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py -> resume_classifier/ -> src/ -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

ENV_PREFIX = "RC_"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Environment(str, Enum):
    """The deployment environment the process is running in."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated, immutable application settings.

    Instances are frozen: configuration is read once at startup and cannot be
    mutated at runtime, which keeps behaviour deterministic and reproducible.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        # Foreign keys (POSTGRES_*, COMPOSE_*) share this file with Docker
        # Compose and must not break the application. The RC_ namespace is
        # still validated strictly, by the checks below.
        extra="ignore",
        frozen=True,
    )

    # -- application ------------------------------------------------------
    app_name: str = "resume-classifier"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = "INFO"

    # -- filesystem -------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"

    # -- dataset build ----------------------------------------------------
    dataset_version: str = "v1"
    random_seed: int = 42
    split_train: float = Field(default=0.70, gt=0.0, lt=1.0)
    split_validation: float = Field(default=0.15, gt=0.0, lt=1.0)
    split_test: float = Field(default=0.15, gt=0.0, lt=1.0)
    min_resume_chars: int = Field(default=200, ge=0)
    min_resume_words: int = Field(default=30, ge=0)
    near_duplicate_threshold: float = Field(default=0.95, gt=0.0, le=1.0)

    # -- inference --------------------------------------------------------
    model_version: str | None = None  # None -> newest version in models_dir
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=40, gt=0)

    # -- database ---------------------------------------------------------
    # Persistence is enabled only when a URL is supplied. The application
    # classifies perfectly well without a database.
    database_url: str | None = None
    database_echo: bool = False
    store_resume_text: bool = False  # privacy default: never persist raw text

    # -- http server ------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = Field(default=5000, gt=0, lt=65536)

    @property
    def is_production(self) -> bool:
        """True when running under the production environment."""
        return self.environment is Environment.PRODUCTION

    @property
    def database_enabled(self) -> bool:
        """Persistence is opt-in: it requires an explicit database URL."""
        return bool(self.database_url)

    @property
    def raw_data_dir(self) -> Path:
        """Immutable source corpora. The application never writes here."""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Derived datasets produced from the raw corpus by the pipeline."""
        return self.data_dir / "processed"

    @property
    def taxonomy_dir(self) -> Path:
        """Curated, version-controlled taxonomies (categories and skills)."""
        return self.data_dir / "taxonomy"

    @property
    def dataset_dir(self) -> Path:
        """Output directory for the active dataset version."""
        return self.processed_data_dir / self.dataset_version

    @model_validator(mode="after")
    def _reject_unknown_environment_variables(self) -> Settings:
        """Fail on a misspelled ``RC_*`` variable instead of using a default.

        pydantic-settings silently ignores prefixed operating system variables
        that match no field, which is the likeliest source of a silent
        misconfiguration: a stray shell export, or a mistyped key in a
        docker-compose ``environment:`` block.
        """
        known = {f"{ENV_PREFIX}{name}".upper() for name in type(self).model_fields}
        unknown = sorted(
            name
            for name in os.environ
            if name.upper().startswith(ENV_PREFIX) and name.upper() not in known
        )
        if unknown:
            msg = f"unknown environment variable(s): {', '.join(unknown)}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _reject_debug_in_production(self) -> Settings:
        """Refuse to start a production process with debug enabled.

        Flask's debug mode exposes an interactive console that is a remote code
        execution vector. Making this a startup failure removes the possibility
        of shipping it by accident.
        """
        if self.is_production and self.debug:
            msg = "RC_DEBUG must be false when RC_ENVIRONMENT=production"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _validate_split_proportions(self) -> Settings:
        """Split proportions must describe a partition of the corpus."""
        total = self.split_train + self.split_validation + self.split_test
        if abs(total - 1.0) > 1e-9:
            msg = f"split proportions must sum to 1.0, got {total!r}"
            raise ValueError(msg)
        return self


def unknown_dotenv_keys(path: Path) -> list[str]:
    """Return ``RC_*`` keys in a dotenv file that are not real settings.

    ``extra="ignore"`` lets Compose share this file, but that also means a
    mistyped ``RC_`` key would be silently discarded. This restores the check
    for the namespace we own, without rejecting the keys we do not.
    """
    if not path.is_file():
        return []
    known = {f"{ENV_PREFIX}{name}".upper() for name in Settings.model_fields}
    unknown: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key.upper().startswith(ENV_PREFIX) and key.upper() not in known:
            unknown.append(key)
    return unknown


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance, constructed once.

    Call ``get_settings.cache_clear()`` to force a reload (used by tests).

    Raises:
        ValueError: if ``.env`` contains a key in the ``RC_`` namespace that is
            not a real setting.
    """
    settings = Settings()
    stray = unknown_dotenv_keys(Path(".env"))
    if stray:
        msg = (
            f"unknown RC_* key(s) in .env: {', '.join(stray)}. "
            "Settings use the RC_ prefix; Docker Compose variables must not."
        )
        raise ValueError(msg)
    return settings
