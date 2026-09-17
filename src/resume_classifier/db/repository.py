"""Data access for analyses.

All SQL knowledge lives here. The service layer asks for what it wants in domain
terms and never sees a query, a session, or an ORM object.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from resume_classifier.db.models import Analysis, AnalysisSkill
from resume_classifier.inference.engine import ClassificationResult

logger = logging.getLogger(__name__)


def text_fingerprint(text: str) -> str:
    """SHA-256 of the document, stored instead of the document itself."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisRepository:
    """Reads and writes classification records."""

    session: Session

    def record(
        self,
        result: ClassificationResult,
        *,
        resume_text: str,
        input_kind: str = "text",
        page_count: int | None = None,
        store_text: bool = False,
    ) -> str:
        """Persist one classification and return its id.

        ``resume_text`` is used to compute the fingerprint. It is only *stored*
        when ``store_text`` is explicitly true.
        """
        analysis = Analysis(
            model_version=result.model_version,
            predicted_label=result.label,
            confidence=result.confidence,
            margin=result.margin,
            is_uncertain=result.is_uncertain,
            text_sha256=text_fingerprint(resume_text),
            text_char_count=result.text_char_count,
            text_word_count=result.text_word_count,
            input_kind=input_kind,
            page_count=page_count,
            resume_text=resume_text if store_text else None,
        )
        analysis.skills = [
            AnalysisSkill(
                skill_id=skill.id,
                category=skill.category,
                occurrences=skill.occurrences,
            )
            for skill in result.skills
        ]
        self.session.add(analysis)
        self.session.flush()
        return analysis.id

    def get(self, analysis_id: str) -> Analysis | None:
        """Fetch one analysis by id."""
        return self.session.get(Analysis, analysis_id)

    def recent(self, limit: int = 20) -> list[Analysis]:
        """Most recent analyses, newest first."""
        statement = select(Analysis).order_by(Analysis.created_at.desc()).limit(limit)
        return list(self.session.scalars(statement))

    def count(self) -> int:
        """Total analyses recorded."""
        return int(self.session.scalar(select(func.count()).select_from(Analysis)) or 0)

    def label_distribution(self) -> dict[str, int]:
        """How many times each label has been predicted."""
        statement = (
            select(Analysis.predicted_label, func.count())
            .group_by(Analysis.predicted_label)
            .order_by(func.count().desc())
        )
        return {
            (label or "uncertain"): int(count) for label, count in self.session.execute(statement)
        }

    def top_skills(self, limit: int = 20) -> list[dict[str, Any]]:
        """Most frequently detected skills across all analyses."""
        statement = (
            select(AnalysisSkill.skill_id, func.count())
            .group_by(AnalysisSkill.skill_id)
            .order_by(func.count().desc())
            .limit(limit)
        )
        return [
            {"skill_id": skill_id, "count": int(count)}
            for skill_id, count in self.session.execute(statement)
        ]
