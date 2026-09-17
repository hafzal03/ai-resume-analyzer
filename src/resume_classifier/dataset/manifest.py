"""Provenance manifests.

Corpora and model artifacts are deliberately excluded from version control --
size, licensing, and personal data all argue against committing them. Manifests
close the resulting gap: they *are* committed, and they record exactly which
bytes produced which outputs.

A manifest answers, for any model in the system: which raw files, at which
hashes, under which taxonomy version, with which seed, at which commit.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def git_commit(repo_root: Path) -> str | None:
    """Return the current commit hash, or ``None`` outside a git checkout.

    Never raises: provenance capture must not break a build on a machine where
    git is unavailable or the project is not a repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a manifest as pretty, sorted, newline-terminated JSON.

    Sorting keys keeps manifests diffable across rebuilds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: Path) -> dict[str, Any]:
    """Read a manifest written by :func:`write_manifest`."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
