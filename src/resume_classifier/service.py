"""Application service layer.

Orchestrates extraction, classification and persistence. It knows nothing about
HTTP -- no request objects, no status codes, no templates -- so the same service
backs the JSON API, the HTML interface, and the tests.

Persistence is best-effort by design: a database failure is logged and reported
in the result, but it never costs the caller their classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from resume_classifier.config import Settings
from resume_classifier.extraction.pdf import (
    ExtractedDocument,
    extract_text_from_bytes,
)
from resume_classifier.inference.engine import ClassificationEngine, ClassificationResult

logger = logging.getLogger(__name__)


class Classifier(Protocol):
    """What the service and the web layer need from a classification engine.

    Declared as a protocol rather than a concrete type so the seam is explicit:
    tests can substitute a stub without loading a model from disk, and nothing
    in the service depends on how the engine was constructed.
    """

    bundle: Any
    taxonomy: Any
    extractor: Any

    @property
    def model_version(self) -> str:
        """Version string of the loaded model."""
        ...

    def classify(self, text: str, *, min_words: int = 30) -> ClassificationResult:
        """Classify one resume."""
        ...


class UploadTooLargeError(ValueError):
    """The uploaded file exceeds the configured size limit."""

    code = "FILE_TOO_LARGE"


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """A completed analysis, plus what happened to it afterwards."""

    result: ClassificationResult
    analysis_id: str | None
    persisted: bool
    document: ExtractedDocument | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form returned by the API."""
        payload = self.result.to_dict()
        payload["analysis_id"] = self.analysis_id
        payload["persisted"] = self.persisted
        if self.document is not None:
            payload["document"]["pages"] = self.document.page_count
            payload["document"]["pages_with_text"] = self.document.pages_with_text
        return payload


@dataclass
class ResumeAnalysisService:
    """Coordinates the full analyse-and-record flow."""

    settings: Settings
    engine: Classifier
    session_factory: Any = None  # sessionmaker[Session] | None

    @property
    def persistence_enabled(self) -> bool:
        """Whether analyses are being recorded."""
        return self.session_factory is not None

    def analyse_text(self, text: str) -> AnalysisOutcome:
        """Classify raw resume text.

        Raises:
            TextTooShortError: when the document is too short to classify.
        """
        result = self.engine.classify(text, min_words=self.settings.min_resume_words)
        analysis_id, persisted = self._persist(result, text, input_kind="text")
        return AnalysisOutcome(result=result, analysis_id=analysis_id, persisted=persisted)

    def analyse_pdf(self, data: bytes) -> AnalysisOutcome:
        """Extract text from PDF bytes, then classify it.

        Raises:
            UploadTooLargeError: when the upload exceeds the size limit.
            PdfExtractionError: when the PDF cannot be turned into text.
            TextTooShortError: when the extracted text is too short.
        """
        if len(data) > self.settings.max_upload_bytes:
            msg = f"file is {len(data):,} bytes; the limit is {self.settings.max_upload_bytes:,}"
            raise UploadTooLargeError(msg)

        document = extract_text_from_bytes(data, max_pages=self.settings.max_pdf_pages)
        result = self.engine.classify(document.text, min_words=self.settings.min_resume_words)

        if document.is_probably_scanned:
            result.warnings.append(
                f"only {document.pages_with_text} of {document.page_count} pages "
                "contained extractable text; the result may be unreliable"
            )

        analysis_id, persisted = self._persist(
            result, document.text, input_kind="pdf", page_count=document.page_count
        )
        return AnalysisOutcome(
            result=result,
            analysis_id=analysis_id,
            persisted=persisted,
            document=document,
        )

    def _persist(
        self,
        result: ClassificationResult,
        text: str,
        *,
        input_kind: str,
        page_count: int | None = None,
    ) -> tuple[str | None, bool]:
        """Record an analysis, tolerating database failure."""
        if self.session_factory is None:
            return None, False

        from resume_classifier.db.repository import AnalysisRepository
        from resume_classifier.db.session import session_scope

        try:
            with session_scope(self.session_factory) as session:
                repository = AnalysisRepository(session)
                analysis_id = repository.record(
                    result,
                    resume_text=text,
                    input_kind=input_kind,
                    page_count=page_count,
                    store_text=self.settings.store_resume_text,
                )
            return analysis_id, True
        except Exception as exc:
            logger.error("failed to persist analysis: %s", exc)
            result.warnings.append("result could not be saved to the database")
            return None, False


def build_service(settings: Settings) -> ResumeAnalysisService:
    """Construct a fully wired service from configuration.

    The database is optional: if it is configured but unreachable, the service
    starts anyway with persistence disabled rather than refusing to serve.
    """
    engine = ClassificationEngine.load(settings)
    session_factory = None

    if settings.database_enabled:
        from resume_classifier.db.session import (
            check_connection,
            create_db_engine,
            create_session_factory,
        )

        try:
            db_engine = create_db_engine(settings)
            if check_connection(db_engine):
                session_factory = create_session_factory(db_engine)
                logger.info("persistence enabled")
            else:
                logger.warning("database unreachable; continuing without persistence")
        except Exception as exc:
            logger.warning("database setup failed (%s); continuing without it", exc)
    else:
        logger.info("RC_DATABASE_URL not set; persistence disabled")

    return ResumeAnalysisService(settings=settings, engine=engine, session_factory=session_factory)
