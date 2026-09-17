"""Shared pytest fixtures.

Tests must never be influenced by the developer's own environment variables or
by a local ``.env`` file, otherwise results differ between machines and CI.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from resume_classifier.config import PROJECT_ROOT, Settings, get_settings
from resume_classifier.logging_config import reset_logging

ENV_PREFIX = "RC_"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip RC_* variables and reset cached global state around every test."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    reset_logging()
    yield
    get_settings.cache_clear()
    reset_logging()


@pytest.fixture
def settings() -> Settings:
    """Default settings, isolated from any local .env file."""
    return Settings(_env_file=None)


@pytest.fixture
def dataset_dir(settings: Settings) -> Path:
    """Location of the built corpus."""
    return settings.dataset_dir


@pytest.fixture
def has_corpus(dataset_dir: Path) -> bool:
    """Whether the dataset has been built on this machine."""
    return (dataset_dir / "corpus.jsonl").is_file()


@pytest.fixture
def taxonomy_dir() -> Path:
    """Directory holding the version-controlled taxonomies."""
    return PROJECT_ROOT / "data" / "taxonomy"


def requires_corpus(dataset_dir: Path) -> None:
    """Skip a test when the corpus has not been built."""
    if not (dataset_dir / "corpus.jsonl").is_file():
        pytest.skip("corpus not built; run 'rc-build-dataset'")


def requires_model(models_dir: Path) -> None:
    """Skip a test when no model has been trained."""
    from resume_classifier.models.artifacts import list_versions

    if not list_versions(models_dir):
        pytest.skip("no trained model; run 'rc-train'")
