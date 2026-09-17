"""End-to-end tests against the real trained model.

These skip cleanly when no model has been trained, so a fresh clone can still
run the suite. When a model *is* present they exercise the whole path:
text in, prediction and skills out, through the same code the web service uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_classifier.config import Settings
from resume_classifier.inference.engine import ClassificationEngine, TextTooShortError
from resume_classifier.models.artifacts import list_versions, load_bundle
from tests.conftest import requires_model

pytestmark = pytest.mark.requires_model


ACCOUNTING_RESUME = """
SENIOR ACCOUNTANT

Summary
Detail-oriented accountant with twelve years of experience in corporate
accounting, month end close and financial reporting.

Experience
Senior Accountant, Meridian Manufacturing
Prepared journal entries and maintained the general ledger. Performed monthly
account reconciliations and variance analysis. Managed accounts payable and
accounts receivable cycles. Supported external audit and prepared tax filings.
Produced financial statements including balance sheet and income statement.
Built budgeting and forecasting models in Microsoft Excel using pivot tables.

Skills
General ledger, journal entries, reconciliations, GAAP, accounts payable,
accounts receivable, budgeting, forecasting, auditing, tax preparation,
QuickBooks, SAP, Microsoft Excel
"""

CULINARY_RESUME = """
EXECUTIVE CHEF

Summary
Executive chef with fifteen years leading high volume restaurant kitchens.

Experience
Executive Chef, Harbour House Restaurant
Directed all kitchen operations and managed a brigade of eighteen cooks.
Designed seasonal menus and developed recipes. Controlled food cost and
inventory. Enforced sanitation standards and HACCP compliance. Trained line
cooks and sous chefs in food preparation techniques.

Skills
Menu development, recipe development, kitchen management, food safety,
ServSafe certified, HACCP, catering, inventory control, team leadership
"""


@pytest.fixture
def engine(settings: Settings) -> ClassificationEngine:
    requires_model(settings.models_dir)
    return ClassificationEngine.load(settings)


class TestArtifacts:
    def test_a_model_exists(self, settings: Settings) -> None:
        requires_model(settings.models_dir)
        assert list_versions(settings.models_dir)

    def test_metadata_records_provenance(self, settings: Settings) -> None:
        requires_model(settings.models_dir)
        bundle = load_bundle(settings.models_dir)
        metadata = bundle.metadata
        assert metadata["labels"]
        assert metadata["sklearn_version"]
        assert metadata["dataset"]["manifest_sha256"]
        assert metadata["training"]["seed"] is not None

    def test_metrics_are_recorded_with_baselines(self, settings: Settings) -> None:
        """A metric without a baseline beside it is uninterpretable."""
        requires_model(settings.models_dir)
        metadata = load_bundle(settings.models_dir).metadata
        assert metadata["metrics"]["test"]["macro_f1"] > 0
        assert metadata["training"]["baselines_cv_macro_f1"]

    def test_model_beats_the_random_baseline_substantially(self, settings: Settings) -> None:
        """The whole point: the model must be doing real work."""
        requires_model(settings.models_dir)
        metadata = load_bundle(settings.models_dir).metadata
        test_f1 = metadata["metrics"]["test"]["macro_f1"]
        baseline = metadata["training"]["baselines_cv_macro_f1"]["baseline_stratified"]
        assert test_f1 > baseline * 5

    def test_every_taxonomy_class_is_predictable(self, settings: Settings) -> None:
        requires_model(settings.models_dir)
        assert len(load_bundle(settings.models_dir).labels) == 24


class TestClassification:
    def test_classifies_an_accounting_resume(self, engine: ClassificationEngine) -> None:
        result = engine.classify(ACCOUNTING_RESUME)
        assert result.label is not None
        # Finance-adjacent classes genuinely overlap; any of them is a sane answer.
        assert result.label in {"accountant", "finance", "banking"}

    def test_classifies_a_culinary_resume(self, engine: ClassificationEngine) -> None:
        result = engine.classify(CULINARY_RESUME)
        assert result.label == "chef"

    def test_extracts_domain_appropriate_skills(self, engine: ClassificationEngine) -> None:
        accounting = {s.id for s in engine.classify(ACCOUNTING_RESUME).skills}
        culinary = {s.id for s in engine.classify(CULINARY_RESUME).skills}
        assert "general_ledger" in accounting
        assert "food_safety" in culinary
        assert accounting != culinary

    def test_probabilities_are_a_distribution(self, engine: ClassificationEngine) -> None:
        result = engine.classify(ACCOUNTING_RESUME)
        assert 0.0 <= result.confidence <= 1.0
        assert all(0.0 <= a["probability"] <= 1.0 for a in result.alternatives)

    def test_alternatives_are_ranked(self, engine: ClassificationEngine) -> None:
        alternatives = engine.classify(ACCOUNTING_RESUME).alternatives
        probabilities = [a["probability"] for a in alternatives]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_short_text_is_refused(self, engine: ClassificationEngine) -> None:
        with pytest.raises(TextTooShortError):
            engine.classify("too short to mean anything")

    def test_classification_is_deterministic(self, engine: ClassificationEngine) -> None:
        """The same document must always produce the same answer."""
        first = engine.classify(ACCOUNTING_RESUME)
        second = engine.classify(ACCOUNTING_RESUME)
        assert first.label == second.label
        assert first.confidence == second.confidence

    def test_result_serialises(self, engine: ClassificationEngine) -> None:
        payload = engine.classify(CULINARY_RESUME).to_dict()
        assert {"prediction", "alternatives", "skills", "model_version"} <= set(payload)

    def test_gibberish_is_handled_without_crashing(self, engine: ClassificationEngine) -> None:
        """Low-signal input must produce a verdict, ideally an uncertain one."""
        result = engine.classify("lorem ipsum dolor sit amet " * 20)
        assert result.display_label


class TestTrainSeveWiring:
    def test_serving_uses_the_same_normalisation_as_training(
        self, engine: ClassificationEngine
    ) -> None:
        """The vectoriser must carry the shared preprocessor inside the artifact.

        If this ever fails, training and serving have drifted apart -- the exact
        defect that made the previous system silently mispredict.
        """
        from resume_classifier.preprocessing.text import feature_normalize

        features = engine.bundle.pipeline.named_steps["features"]
        vectorizers = (
            [v for _, v in features.transformer_list]
            if hasattr(features, "transformer_list")
            else [features]
        )
        assert vectorizers
        for vectorizer in vectorizers:
            assert vectorizer.preprocessor is feature_normalize


class TestApiWithRealModel:
    def test_full_http_path(self, settings: Settings, tmp_path: Path) -> None:
        requires_model(settings.models_dir)
        from resume_classifier.api.app import create_app
        from resume_classifier.service import build_service

        app = create_app(settings, service=build_service(settings))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.post("/api/v1/classify/text", json={"text": CULINARY_RESUME})
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["prediction"]["label"] == "chef"
        assert payload["skills"]
        assert payload["persisted"] is False  # no database configured in tests
