"""Source adapters: turning a third-party corpus into raw rows.

An adapter knows one corpus's quirks -- its filename, its column names, its
label spelling -- and nothing else. Everything downstream (validation,
deduplication, splitting) is corpus-agnostic, so supporting a second corpus
means writing one adapter, not touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class IngestError(RuntimeError):
    """The source corpus is missing, unreadable, or has an unexpected shape."""


@dataclass(frozen=True, slots=True)
class RawRow:
    """One row as the source corpus provides it, before any validation."""

    source_id: str
    text: str
    label_raw: str


class SourceAdapter(Protocol):
    """Reads one corpus and yields :class:`RawRow` values."""

    @property
    def source(self) -> str:
        """Short, stable corpus identifier used in record ids."""
        ...

    @property
    def source_version(self) -> str:
        """Snapshot identifier of the corpus."""
        ...

    def rows(self) -> Iterator[RawRow]:
        """Yield every row in the corpus, in a deterministic order."""
        ...


@dataclass(frozen=True, slots=True)
class LiveCareerAdapter:
    """Adapter for the LiveCareer resume corpus.

    Expects ``Resume.csv`` with columns ``ID``, ``Resume_str``, ``Resume_html``
    and ``Category``. The HTML column is deliberately ignored: it carries the
    same content as the text column plus markup that would pollute the feature
    space.
    """

    root: Path
    source: str = "livecareer_resumes"
    source_version: str = "v1"

    @property
    def csv_path(self) -> Path:
        """Location of the corpus CSV."""
        return self.root / "Resume.csv"

    def rows(self) -> Iterator[RawRow]:
        """Yield corpus rows ordered by source id for determinism."""
        import pandas as pd

        if not self.csv_path.is_file():
            msg = f"corpus not found at {self.csv_path}. See docs/dataset.md for how to obtain it."
            raise IngestError(msg)

        frame = pd.read_csv(self.csv_path)
        required = {"ID", "Resume_str", "Category"}
        missing = required - set(frame.columns)
        if missing:
            msg = f"{self.csv_path} is missing column(s): {sorted(missing)}"
            raise IngestError(msg)

        frame = frame.sort_values("ID", kind="stable")
        for row in frame.itertuples(index=False):
            yield RawRow(
                source_id=str(row.ID),
                text="" if row.Resume_str is None else str(row.Resume_str),
                label_raw="" if row.Category is None else str(row.Category),
            )


def get_adapter(name: str, raw_root: Path) -> SourceAdapter:
    """Return the adapter registered under ``name``.

    Raises:
        IngestError: if no adapter is registered for that name.
    """
    if name == "livecareer_resumes":
        return LiveCareerAdapter(root=raw_root / name / "v1")
    msg = f"no ingest adapter registered for source {name!r}"
    raise IngestError(msg)
