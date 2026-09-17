"""HTTP routes.

Two surfaces share one service: a JSON API under ``/api/v1`` and a small HTML
interface at ``/``. Neither contains business logic -- they validate input, call
the service, and render the result.
"""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from flask import Blueprint, current_app, jsonify, render_template, request

from resume_classifier.api.errors import ApiError
from resume_classifier.service import ResumeAnalysisService

logger = logging.getLogger(__name__)

api = Blueprint("api", __name__, url_prefix="/api/v1")
web = Blueprint("web", __name__)


def _service() -> ResumeAnalysisService:
    service: ResumeAnalysisService = current_app.extensions["resume_service"]
    return service


# ===========================================================================
# JSON API
# ===========================================================================
@api.get("/health")
def health() -> Any:
    """Liveness: the process is up. Deliberately checks no dependencies."""
    return jsonify({"status": "ok"})


@api.get("/ready")
def ready() -> Any:
    """Readiness: the model is loaded and, if configured, the database answers."""
    service = _service()
    checks: dict[str, Any] = {
        "model_loaded": True,
        "model_version": service.engine.model_version,
        "persistence_enabled": service.persistence_enabled,
    }
    return jsonify({"status": "ready", "checks": checks})


@api.get("/model")
def model_info() -> Any:
    """Active model version, metrics and provenance."""
    bundle = _service().engine.bundle
    metrics = bundle.metadata.get("metrics", {})
    return jsonify(
        {
            "version": bundle.version,
            "created_at": bundle.metadata.get("created_at"),
            "estimator": bundle.metadata.get("selected", {}).get("estimator"),
            "features": bundle.metadata.get("selected", {}).get("features", {}).get("name"),
            "hyperparameters": bundle.metadata.get("selected", {}).get("hyperparameters"),
            "labels": bundle.labels,
            "label_count": len(bundle.labels),
            "confidence": {
                "threshold": bundle.confidence_threshold,
                "margin": bundle.margin_threshold,
            },
            "metrics": {
                "test": metrics.get("test", {}).get("macro_f1"),
                "test_accuracy": metrics.get("test", {}).get("accuracy"),
                "validation": metrics.get("validation", {}).get("macro_f1"),
            },
            "baselines": bundle.metadata.get("training", {}).get("baselines_cv_macro_f1"),
            "dataset": bundle.metadata.get("dataset"),
            "sklearn_version": bundle.metadata.get("sklearn_version"),
        }
    )


@api.get("/categories")
def categories() -> Any:
    """The closed set of classes this model can predict."""
    taxonomy = _service().engine.taxonomy
    return jsonify(
        {
            "taxonomy_version": taxonomy.version,
            "count": len(taxonomy.classes),
            "categories": [
                {"id": c.id, "display": c.display, "description": c.description}
                for c in taxonomy.classes
            ],
        }
    )


@api.get("/skills")
def skills() -> Any:
    """The skill vocabulary used for extraction."""
    extractor = _service().engine.extractor
    return jsonify(
        {
            "taxonomy_version": extractor.taxonomy.version,
            "count": len(extractor.skill_ids),
            "categories": list(extractor.taxonomy.categories),
            "skills": [
                {"id": s.id, "display": s.display, "category": s.category}
                for s in extractor.taxonomy.skills
            ],
        }
    )


@api.post("/classify")
def classify_pdf() -> Any:
    """Classify an uploaded PDF (multipart form field ``resume``)."""
    uploaded = request.files.get("resume")
    if uploaded is None or not uploaded.filename:
        msg = "no file supplied; attach a PDF under the form field 'resume'"
        raise ApiError(msg, code="NO_FILE", status=HTTPStatus.BAD_REQUEST)

    outcome = _service().analyse_pdf(uploaded.read())
    return jsonify(outcome.to_dict())


@api.post("/classify/text")
def classify_text() -> Any:
    """Classify raw text supplied as JSON ``{"text": "..."}``.

    Exists so the whole pipeline is testable and scriptable without constructing
    a PDF.
    """
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        msg = 'request body must be JSON of the form {"text": "..."}'
        raise ApiError(msg, code="INVALID_BODY", status=HTTPStatus.BAD_REQUEST)

    outcome = _service().analyse_text(payload["text"])
    return jsonify(outcome.to_dict())


# ===========================================================================
# HTML interface
# ===========================================================================
@web.get("/")
def index() -> Any:
    """Upload form."""
    service = _service()
    return render_template(
        "index.html",
        model_version=service.engine.model_version,
        category_count=len(service.engine.taxonomy.classes),
        skill_count=len(service.engine.extractor.skill_ids),
    )


@web.post("/")
def analyse() -> Any:
    """Handle the upload form and render the result."""
    service = _service()
    uploaded = request.files.get("resume")
    pasted = (request.form.get("resume_text") or "").strip()

    if uploaded is not None and uploaded.filename:
        outcome = service.analyse_pdf(uploaded.read())
    elif pasted:
        outcome = service.analyse_text(pasted)
    else:
        msg = "upload a PDF or paste resume text"
        raise ApiError(msg, code="NO_INPUT", status=HTTPStatus.BAD_REQUEST)

    return render_template(
        "result.html",
        outcome=outcome,
        result=outcome.result,
        model_version=service.engine.model_version,
    )
