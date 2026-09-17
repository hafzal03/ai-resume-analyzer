"""A single error envelope for the whole API.

Every failure -- validation, extraction, missing model, unexpected crash --
leaves through here in the same JSON shape, with a machine-readable ``code``
and a message safe to show a user. Stack traces never reach a client.
"""

from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from resume_classifier.extraction.pdf import (
    EncryptedPdfError,
    NotAPdfError,
    NoTextLayerError,
    PdfExtractionError,
    TooManyPagesError,
)
from resume_classifier.inference.engine import TextTooShortError
from resume_classifier.models.artifacts import ModelNotFoundError
from resume_classifier.service import UploadTooLargeError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An error with an explicit HTTP status and code."""

    def __init__(self, message: str, *, code: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def error_payload(code: str, message: str, *, request_id: str) -> dict[str, Any]:
    """The canonical error body."""
    return {
        "error": {"code": code, "message": message},
        "request_id": request_id,
    }


def _wants_json() -> bool:
    """Whether to answer with JSON rather than an HTML page."""
    if request.path.startswith("/api/"):
        return True
    return request.accept_mimetypes.best == "application/json"


def _respond(code: str, message: str, status: int) -> Any:
    request_id = str(uuid.uuid4())[:8]
    logger.info("request failed code=%s status=%s request_id=%s", code, status, request_id)
    if _wants_json():
        return jsonify(error_payload(code, message, request_id=request_id)), status
    return (
        render_template("error.html", code=code, message=message, request_id=request_id),
        status,
    )


def register_error_handlers(app: Flask) -> None:
    """Attach handlers mapping every known failure to a status code."""

    @app.errorhandler(ApiError)
    def _api_error(exc: ApiError) -> Any:
        return _respond(exc.code, exc.message, exc.status)

    @app.errorhandler(TextTooShortError)
    def _text_too_short(exc: TextTooShortError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)

    @app.errorhandler(NotAPdfError)
    def _not_a_pdf(exc: NotAPdfError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.UNSUPPORTED_MEDIA_TYPE)

    @app.errorhandler(EncryptedPdfError)
    def _encrypted(exc: EncryptedPdfError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)

    @app.errorhandler(TooManyPagesError)
    def _too_many_pages(exc: TooManyPagesError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    @app.errorhandler(NoTextLayerError)
    def _no_text(exc: NoTextLayerError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)

    @app.errorhandler(PdfExtractionError)
    def _pdf_error(exc: PdfExtractionError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.UNPROCESSABLE_ENTITY)

    @app.errorhandler(UploadTooLargeError)
    def _upload_too_large(exc: UploadTooLargeError) -> Any:
        return _respond(exc.code, str(exc), HTTPStatus.REQUEST_ENTITY_TOO_LARGE)

    @app.errorhandler(RequestEntityTooLarge)
    def _werkzeug_too_large(_: RequestEntityTooLarge) -> Any:
        return _respond(
            "FILE_TOO_LARGE",
            "the uploaded file exceeds the configured size limit",
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )

    @app.errorhandler(ModelNotFoundError)
    def _no_model(exc: ModelNotFoundError) -> Any:
        return _respond("MODEL_UNAVAILABLE", str(exc), HTTPStatus.SERVICE_UNAVAILABLE)

    @app.errorhandler(HTTPException)
    def _http_exception(exc: HTTPException) -> Any:
        return _respond(
            (exc.name or "HTTP_ERROR").upper().replace(" ", "_"),
            exc.description or "request failed",
            exc.code or HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception) -> Any:
        # Log the detail; show the client nothing but a reference.
        logger.exception("unhandled error: %s", exc)
        return _respond(
            "INTERNAL_ERROR",
            "an unexpected error occurred while processing the request",
            HTTPStatus.INTERNAL_SERVER_ERROR,
        )
