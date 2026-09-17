"""Tests for the persistence layer.

These run against an in-memory SQLite database so the suite needs no PostgreSQL.
That is a deliberate trade: the ORM mappings, the repository logic and the
privacy defaults are exercised everywhere, while PostgreSQL-specific behaviour
is covered by the Alembic migration and the compose stack.

The privacy assertions are the important ones here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from resume_classifier.db.models import Analysis, Base
from resume_classifier.db.repository import AnalysisRepository, text_fingerprint
from resume_classifier.inference.engine import ClassificationResult
from resume_classifier.skills.extractor import ExtractedSkill

RESUME_TEXT = "Senior accountant with general ledger and reconciliation experience."


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory() as active:
        yield active


@pytest.fixture
def repository(session: Session) -> AnalysisRepository:
    return AnalysisRepository(session)


def _result(label: str | None = "accountant", uncertain: bool = False) -> ClassificationResult:
    return ClassificationResult(
        label=label,
        display_label="Accountant" if label else "Uncertain",
        confidence=0.87,
        margin=0.4,
        is_uncertain=uncertain,
        alternatives=[],
        skills=[
            ExtractedSkill(
                id="general_ledger",
                display="General Ledger",
                category="finance",
                matched_alias="general ledger",
                occurrences=3,
            ),
            ExtractedSkill(
                id="microsoft_excel",
                display="Microsoft Excel",
                category="professional",
                matched_alias="excel",
                occurrences=1,
            ),
        ],
        model_version="v1",
        text_char_count=len(RESUME_TEXT),
        text_word_count=len(RESUME_TEXT.split()),
    )


class TestPrivacyDefaults:
    def test_resume_text_is_not_stored_by_default(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        """The whole point of the schema: record the outcome, not the document."""
        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert stored.resume_text is None

    def test_text_is_stored_only_when_explicitly_enabled(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT, store_text=True)
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert stored.resume_text == RESUME_TEXT

    def test_fingerprint_is_recorded_instead(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert stored.text_sha256 == text_fingerprint(RESUME_TEXT)

    def test_identical_documents_share_a_fingerprint(self) -> None:
        assert text_fingerprint("abc") == text_fingerprint("abc")
        assert text_fingerprint("abc") != text_fingerprint("abd")

    def test_schema_has_no_protected_characteristic_columns(self) -> None:
        """No age, gender, race, or date-of-birth column may ever exist here."""
        columns = set(Analysis.__table__.columns.keys())
        forbidden = {"age", "gender", "race", "ethnicity", "date_of_birth", "dob"}
        assert columns & forbidden == set()

    def test_schema_does_not_store_the_filename(self) -> None:
        """Uploaded filenames routinely contain candidate names."""
        columns = set(Analysis.__table__.columns.keys())
        assert "filename" not in columns
        assert "original_filename" not in columns


class TestRepository:
    def test_record_returns_an_id(self, repository: AnalysisRepository) -> None:
        assert repository.record(_result(), resume_text=RESUME_TEXT)

    def test_skills_are_persisted(self, repository: AnalysisRepository, session: Session) -> None:
        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert {s.skill_id for s in stored.skills} == {"general_ledger", "microsoft_excel"}

    def test_model_version_is_recorded(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        """Without provenance a past prediction cannot be reproduced."""
        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert stored.model_version == "v1"

    def test_uncertain_results_store_a_null_label(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        analysis_id = repository.record(
            _result(label=None, uncertain=True), resume_text=RESUME_TEXT
        )
        session.commit()
        stored = session.get(Analysis, analysis_id)
        assert stored is not None
        assert stored.predicted_label is None
        assert stored.is_uncertain is True

    def test_count_and_recent(self, repository: AnalysisRepository, session: Session) -> None:
        for _ in range(3):
            repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        assert repository.count() == 3
        assert len(repository.recent(limit=2)) == 2

    def test_label_distribution(self, repository: AnalysisRepository, session: Session) -> None:
        repository.record(_result(label="accountant"), resume_text=RESUME_TEXT)
        repository.record(_result(label="chef"), resume_text="chef text")
        repository.record(_result(label="chef"), resume_text="other chef text")
        session.commit()
        assert repository.label_distribution()["chef"] == 2

    def test_top_skills(self, repository: AnalysisRepository, session: Session) -> None:
        for _ in range(2):
            repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        top = repository.top_skills(limit=5)
        assert {row["skill_id"] for row in top} == {"general_ledger", "microsoft_excel"}

    def test_deleting_an_analysis_removes_its_skills(
        self, repository: AnalysisRepository, session: Session
    ) -> None:
        """Retention deletion must not leave orphaned skill rows behind."""
        from resume_classifier.db.models import AnalysisSkill

        analysis_id = repository.record(_result(), resume_text=RESUME_TEXT)
        session.commit()
        session.delete(session.get(Analysis, analysis_id))
        session.commit()
        assert session.query(AnalysisSkill).count() == 0
