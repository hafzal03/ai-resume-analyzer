"""Tests for PDF extraction.

Every failure mode here corresponds to something a real upload does: a file that
is not a PDF, a scan with no text layer, a page that returns ``None``. The old
implementation crashed with a ``TypeError`` on the last of those.
"""

from __future__ import annotations

import zlib

import pytest

from resume_classifier.extraction.pdf import (
    NotAPdfError,
    NoTextLayerError,
    PdfExtractionError,
    TooManyPagesError,
    extract_text_from_bytes,
    looks_like_pdf,
)


def _minimal_pdf(text: str, pages: int = 1) -> bytes:
    """Build a tiny but valid single-font PDF containing ``text`` on each page."""
    objects: list[bytes] = []

    def stream_for(body: str) -> bytes:
        content = f"BT /F1 12 Tf 72 720 Td ({body}) Tj ET".encode("latin-1", "replace")
        return b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content)

    kids = " ".join(f"{4 + i * 2} 0 R" for i in range(pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Count {pages} /Kids [{kids}] >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i in range(pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + i * 2} 0 R >>".encode("latin-1")
        )
        objects.append(stream_for(text))

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


class TestMagicBytes:
    def test_recognises_a_pdf(self) -> None:
        assert looks_like_pdf(b"%PDF-1.7 rest")

    def test_rejects_other_content(self) -> None:
        """Extension and MIME headers are attacker-controlled; magic bytes are not."""
        assert not looks_like_pdf(b"PK\x03\x04 zipfile")
        assert not looks_like_pdf(b"<html>")


class TestExtraction:
    def test_empty_upload_is_rejected(self) -> None:
        with pytest.raises(NotAPdfError, match="empty"):
            extract_text_from_bytes(b"")

    def test_non_pdf_is_rejected(self) -> None:
        with pytest.raises(NotAPdfError):
            extract_text_from_bytes(b"just some text, definitely not a pdf")

    def test_renamed_zip_is_rejected(self) -> None:
        with pytest.raises(NotAPdfError):
            extract_text_from_bytes(b"PK\x03\x04" + zlib.compress(b"payload"))

    def test_corrupt_pdf_raises_a_typed_error(self) -> None:
        with pytest.raises(PdfExtractionError):
            extract_text_from_bytes(b"%PDF-1.4\nbut then garbage")

    def test_extracts_text(self) -> None:
        document = extract_text_from_bytes(_minimal_pdf("Senior Python Engineer"))
        assert "Python" in document.text
        assert document.page_count == 1
        assert document.pages_with_text == 1

    def test_counts_pages(self) -> None:
        document = extract_text_from_bytes(_minimal_pdf("Resume content", pages=3))
        assert document.page_count == 3

    def test_page_limit_is_enforced(self) -> None:
        """An unbounded page count is a denial-of-service vector."""
        with pytest.raises(TooManyPagesError):
            extract_text_from_bytes(_minimal_pdf("x", pages=5), max_pages=2)

    def test_pdf_without_text_layer_is_reported_not_guessed(self) -> None:
        """A scan must be refused, never classified from an empty string."""
        with pytest.raises(NoTextLayerError):
            extract_text_from_bytes(_minimal_pdf("   "))
