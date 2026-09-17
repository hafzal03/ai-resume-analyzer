"""Package-level integrity checks."""

from __future__ import annotations

import tomllib

import resume_classifier
from resume_classifier.config import PROJECT_ROOT


def test_version_matches_pyproject() -> None:
    """Guard against the package version drifting from the packaging metadata."""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert resume_classifier.__version__ == pyproject["project"]["version"]


def test_project_root_is_resolved_correctly() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "resume_classifier").is_dir()
