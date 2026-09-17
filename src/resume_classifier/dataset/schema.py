"""Canonical dataset record schema and validation vocabulary.

One row of the corpus is one resume. ``label_raw`` is preserved verbatim next to
the canonical ``label`` so every taxonomy mapping decision stays auditable and
reversible.

Records that fail validation are never dropped silently: they are written to a
quarantine file carrying a :class:`RejectReason`, and the build reconciles
``total_in == kept + quarantined``.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from resume_classifier.preprocessing.text import feature_normalize


class Split(str, Enum):
    """The three partitions of the corpus."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RejectReason(str, Enum):
    """Why a source row did not make it into the corpus."""

    TEXT_EMPTY = "TEXT_EMPTY"
    TEXT_TOO_SHORT = "TEXT_TOO_SHORT"
    LABEL_MISSING = "LABEL_MISSING"
    LABEL_UNMAPPED = "LABEL_UNMAPPED"
    ID_DUPLICATE = "ID_DUPLICATE"
    EXACT_DUPLICATE = "EXACT_DUPLICATE"
    NEAR_DUPLICATE = "NEAR_DUPLICATE"


class ResumeRecord(BaseModel):
    """A validated corpus record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_id: str = Field(min_length=1)

    resume_text: str = Field(min_length=1)
    label: str = Field(min_length=1)
    label_raw: str = Field(min_length=1)

    text_sha256: str = Field(min_length=64, max_length=64)
    text_length_chars: int = Field(ge=0)
    text_length_words: int = Field(ge=0)

    duplicate_group_id: str | None = None
    ingested_at: str = Field(min_length=1)


class RejectedRecord(BaseModel):
    """A source row that failed the validation gate, with the reason why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    source_id: str
    reason: RejectReason
    detail: str = ""
    label_raw: str | None = None
    text_length_chars: int = 0


def make_record_id(source: str, source_id: str) -> str:
    """Derive a stable primary key from provenance, not from row order.

    The same source row always produces the same id, so rebuilding the corpus
    never renumbers anything and split assignments stay valid across rebuilds.
    """
    digest = hashlib.sha256(f"{source}::{source_id}".encode()).hexdigest()
    return digest[:16]


def content_hash(text: str) -> str:
    """Hash aggressively normalised text -- the exact-duplicate key.

    Using the feature-level normalisation means formatting-only differences
    (case, punctuation, whitespace) collapse to the same hash.
    """
    return hashlib.sha256(feature_normalize(text).encode("utf-8")).hexdigest()
