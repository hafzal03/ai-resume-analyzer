"""Database schema.

The guiding rule is data minimisation. A resume contains a person's name,
address, telephone number, e-mail and employment history; none of that is needed
to answer "what did the model predict, with which version, and can we reproduce
it". So the default schema stores a **content hash** rather than the document.

Storing the text at all is opt-in via ``RC_STORE_RESUME_TEXT`` and off by
default. There is deliberately no column for the uploaded filename (filenames
routinely contain candidate names) and none for age or any other protected
characteristic.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for every mapped class."""


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Analysis(Base):
    """One classification event."""

    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    # -- provenance: without this, a past prediction is unreproducible ----
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)

    # -- outcome ----------------------------------------------------------
    predicted_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    margin: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_uncertain: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # -- document facts, not document content -----------------------------
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    text_char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    text_word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_kind: Mapped[str] = mapped_column(String(16), nullable=False, default="text")
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # -- opt-in only, off by default --------------------------------------
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    skills: Mapped[list[AnalysisSkill]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("ix_analyses_created_at", "created_at"),
        Index("ix_analyses_model_version", "model_version"),
        Index("ix_analyses_predicted_label", "predicted_label"),
        Index("ix_analyses_text_sha256", "text_sha256"),
    )


class AnalysisSkill(Base):
    """A skill detected in one analysis.

    Stores the canonical taxonomy id, never the raw matched string, so the table
    stays joinable against a versioned vocabulary.
    """

    __tablename__ = "analysis_skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    skill_id: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    analysis: Mapped[Analysis] = relationship(back_populates="skills")

    __table_args__ = (
        Index("ix_analysis_skills_analysis_id", "analysis_id"),
        Index("ix_analysis_skills_skill_id", "skill_id"),
    )
