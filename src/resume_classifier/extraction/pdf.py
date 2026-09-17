"""PDF text extraction.

Real uploads are hostile in mundane ways: pages that yield ``None`` instead of
text, scanned documents with no text layer at all, encrypted files, and PDFs
whose page count is a denial-of-service vector. Each of those is handled
explicitly and turned into a typed error the API can map to a status code,
rather than a 500.

``pypdf`` is used rather than the deprecated ``PyPDF2``. No OCR is performed:
a scanned resume is reported as unextractable instead of silently classified
from an empty string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from resume_classifier.preprocessing.text import ingestion_normalize, word_count

logger = logging.getLogger(__name__)

#: Magic bytes every PDF begins with.
PDF_MAGIC = b"%PDF-"


class PdfExtractionError(ValueError):
    """The upload could not be turned into usable text."""

    #: Short machine-readable code, surfaced to API clients.
    code = "PDF_EXTRACTION_FAILED"


class NotAPdfError(PdfExtractionError):
    """The bytes supplied are not a PDF."""

    code = "NOT_A_PDF"


class EncryptedPdfError(PdfExtractionError):
    """The PDF is password protected."""

    code = "PDF_ENCRYPTED"


class TooManyPagesError(PdfExtractionError):
    """The PDF exceeds the configured page limit."""

    code = "PDF_TOO_MANY_PAGES"


class NoTextLayerError(PdfExtractionError):
    """The PDF parsed, but contains no extractable text (likely a scan)."""

    code = "PDF_NO_TEXT"


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """Text recovered from a PDF, plus what was learned along the way."""

    text: str
    page_count: int
    pages_with_text: int
    char_count: int
    word_count: int

    @property
    def is_probably_scanned(self) -> bool:
        """True when pages parsed but almost none yielded text."""
        return self.page_count > 0 and self.pages_with_text <= self.page_count // 4


def looks_like_pdf(data: bytes) -> bool:
    """Check the magic bytes rather than trusting a filename or MIME header."""
    return data[:5] == PDF_MAGIC


def extract_text_from_bytes(data: bytes, *, max_pages: int = 40) -> ExtractedDocument:
    """Extract text from PDF bytes.

    Args:
        data: Raw file content.
        max_pages: Refuse documents longer than this.

    Raises:
        NotAPdfError: The bytes are not a PDF.
        EncryptedPdfError: The document is password protected.
        TooManyPagesError: The document exceeds ``max_pages``.
        NoTextLayerError: The document has no extractable text.
        PdfExtractionError: The document is structurally unreadable.
    """
    if not data:
        msg = "uploaded file is empty"
        raise NotAPdfError(msg)
    if not looks_like_pdf(data):
        msg = "file does not begin with the PDF magic bytes"
        raise NotAPdfError(msg)

    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        msg = f"could not parse PDF: {exc}"
        raise PdfExtractionError(msg) from exc

    if reader.is_encrypted:
        # An empty-password decrypt succeeds surprisingly often; try once.
        try:
            if reader.decrypt("") == 0:  # 0 == failure
                msg = "PDF is password protected"
                raise EncryptedPdfError(msg)
        except (NotImplementedError, PdfReadError) as exc:
            msg = f"PDF uses unsupported encryption: {exc}"
            raise EncryptedPdfError(msg) from exc

    page_count = len(reader.pages)
    if page_count == 0:
        msg = "PDF contains no pages"
        raise PdfExtractionError(msg)
    if page_count > max_pages:
        msg = f"PDF has {page_count} pages, limit is {max_pages}"
        raise TooManyPagesError(msg)

    parts: list[str] = []
    pages_with_text = 0
    for index, page in enumerate(reader.pages):
        try:
            # extract_text() legitimately returns None for image-only pages.
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("page %d failed to extract: %s", index, exc)
            page_text = ""
        if page_text.strip():
            pages_with_text += 1
            parts.append(page_text)

    text = ingestion_normalize("\n".join(parts))
    if not text:
        msg = "no extractable text found; the document is probably a scan or image-only PDF"
        raise NoTextLayerError(msg)

    return ExtractedDocument(
        text=text,
        page_count=page_count,
        pages_with_text=pages_with_text,
        char_count=len(text),
        word_count=word_count(text),
    )


def extract_text_from_path(path: Path, *, max_pages: int = 40) -> ExtractedDocument:
    """Extract text from a PDF on disk."""
    return extract_text_from_bytes(path.read_bytes(), max_pages=max_pages)
