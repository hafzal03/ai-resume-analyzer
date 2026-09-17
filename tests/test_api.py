"""Tests for the HTTP layer.

A stub engine is injected so these run without a trained model on disk: the
routes, the error envelope and the status-code mapping are what is under test,
not the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from flask.testing import FlaskClient

from resume_classifier.api.app import create_app
from resume_classifier.config import Environment, Settings
from resume_classifier.dataset.taxonomy import load_taxonomy
from resume_classifier.inference.engine import ClassificationResult, TextTooShortError
from resume_classifier.service import ResumeAnalysisService
from resume_classifier.skills.extractor import SkillExtractor
from resume_classifier.skills.taxonomy import load_skill_taxonomy


@dataclass
class _StubBundle:
    version: str = "vtest"
    metadata: dict[str, Any] | None = None
    labels: list[str] | None = None

    def __post_init__(self) -> None:
        self.metadata = self.metadata or {
            "created_at": "2026-01-01T00:00:00+00:00",
            "selected": {"estimator": "linear_svc", "features": {"name": "word_1gram"}},
            "metrics": {"test": {"macro_f1": 0.6, "accuracy": 0.65}},
            "sklearn_version": "1.9.1",
        }
        self.labels = self.labels or ["accountant", "chef"]

    confidence_threshold: float = 0.0
    margin_threshold: float = 0.1


@dataclass
class _StubEngine:
    taxonomy: Any
    extractor: SkillExtractor
    bundle: _StubBundle
    model_version: str = "vtest"

    def classify(self, text: str, *, min_words: int = 30) -> ClassificationResult:
        if len(text.split()) < min_words:
            msg = f"document has {len(text.split())} usable words"
            raise TextTooShortError(msg)
        return ClassificationResult(
            label="accountant",
            display_label="Accountant",
            confidence=0.91,
            margin=0.4,
            is_uncertain=False,
            alternatives=[{"label": "accountant", "display": "Accountant", "probability": 0.91}],
            skills=self.extractor.extract(text),
            model_version=self.model_version,
            text_char_count=len(text),
            text_word_count=len(text.split()),
            warnings=[],
        )


@pytest.fixture
def client(taxonomy_dir: Any) -> FlaskClient:
    settings = Settings(_env_file=None, environment=Environment.TESTING)
    engine = _StubEngine(
        taxonomy=load_taxonomy(taxonomy_dir / "categories_v1.toml"),
        extractor=SkillExtractor(load_skill_taxonomy(taxonomy_dir / "skills_v1.toml")),
        bundle=_StubBundle(),
    )
    service = ResumeAnalysisService(settings=settings, engine=engine, session_factory=None)
    app = create_app(settings, service=service)
    app.config["TESTING"] = True
    return app.test_client()


LONG_TEXT = (
    "Senior accountant with ten years of experience in general ledger, "
    "accounts payable, accounts receivable, month end close, financial "
    "reporting, budgeting, forecasting, audit support and tax preparation. "
    "Skilled in Microsoft Excel, QuickBooks and SAP. Managed a team of four."
)


class TestHealthAndMetadata:
    def test_health_is_dependency_free(self, client: FlaskClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"

    def test_ready_reports_model_state(self, client: FlaskClient) -> None:
        response = client.get("/api/v1/ready")
        assert response.status_code == 200
        assert response.get_json()["checks"]["model_loaded"] is True

    def test_model_endpoint_exposes_provenance(self, client: FlaskClient) -> None:
        payload = client.get("/api/v1/model").get_json()
        assert payload["version"] == "vtest"
        assert payload["estimator"] == "linear_svc"

    def test_categories_endpoint(self, client: FlaskClient) -> None:
        payload = client.get("/api/v1/categories").get_json()
        assert payload["count"] == 24
        assert all("id" in c and "display" in c for c in payload["categories"])

    def test_skills_endpoint(self, client: FlaskClient) -> None:
        payload = client.get("/api/v1/skills").get_json()
        assert payload["count"] > 100


class TestClassifyText:
    def test_happy_path(self, client: FlaskClient) -> None:
        response = client.post("/api/v1/classify/text", json={"text": LONG_TEXT})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["prediction"]["label"] == "accountant"
        assert payload["skills"]

    def test_missing_body_is_a_400(self, client: FlaskClient) -> None:
        response = client.post("/api/v1/classify/text", json={})
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "INVALID_BODY"

    def test_wrong_type_is_a_400(self, client: FlaskClient) -> None:
        response = client.post("/api/v1/classify/text", json={"text": 123})
        assert response.status_code == 400

    def test_short_text_is_a_422(self, client: FlaskClient) -> None:
        response = client.post("/api/v1/classify/text", json={"text": "too short"})
        assert response.status_code == 422
        assert response.get_json()["error"]["code"] == "TEXT_TOO_SHORT"


class TestClassifyPdf:
    def test_missing_file_is_a_400(self, client: FlaskClient) -> None:
        response = client.post("/api/v1/classify", data={})
        assert response.status_code == 400
        assert response.get_json()["error"]["code"] == "NO_FILE"

    def test_non_pdf_is_a_415(self, client: FlaskClient) -> None:
        from io import BytesIO

        response = client.post(
            "/api/v1/classify",
            data={"resume": (BytesIO(b"not a pdf at all"), "evil.pdf")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 415
        assert response.get_json()["error"]["code"] == "NOT_A_PDF"


class TestErrorEnvelope:
    def test_unknown_route_returns_the_standard_shape(self, client: FlaskClient) -> None:
        payload = client.get("/api/v1/nope").get_json()
        assert "error" in payload
        assert {"code", "message"} <= set(payload["error"])
        assert "request_id" in payload

    def test_errors_never_leak_a_traceback(self, client: FlaskClient) -> None:
        body = client.post("/api/v1/classify/text", json={}).get_data(as_text=True)
        assert "Traceback" not in body
        assert 'File "' not in body


class TestWebInterface:
    def test_index_renders(self, client: FlaskClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert b"Resume Classifier" in response.data

    def test_form_submission_renders_a_result(self, client: FlaskClient) -> None:
        response = client.post("/", data={"resume_text": LONG_TEXT})
        assert response.status_code == 200
        assert b"Accountant" in response.data

    def test_empty_submission_is_rejected(self, client: FlaskClient) -> None:
        response = client.post("/", data={"resume_text": "  "})
        assert response.status_code == 400
