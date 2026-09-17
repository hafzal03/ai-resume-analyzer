"""Tests for feature construction and metric computation.

These are the pure parts of the training pipeline. Covering them here means a
regression is caught in seconds rather than twenty minutes into a training run.
"""

from __future__ import annotations

import pytest
from sklearn.pipeline import FeatureUnion

from resume_classifier.features.vectorizer import (
    FEATURE_CONFIGS,
    FeatureConfig,
    build_vectorizer,
)
from resume_classifier.models.evaluate import (
    confusion_matrix_dict,
    coverage_accuracy_curve,
    evaluate_predictions,
    text_classification_report,
    top_confusions,
)
from resume_classifier.preprocessing.text import feature_normalize

LABELS = ["accountant", "chef", "engineering"]


# ===========================================================================
# Feature construction
# ===========================================================================
class TestVectorizer:
    def test_word_only_config_returns_a_single_vectorizer(self) -> None:
        vectorizer = build_vectorizer(FeatureConfig(name="w", word_ngram_max=1))
        assert not isinstance(vectorizer, FeatureUnion)

    def test_char_config_returns_a_union(self) -> None:
        vectorizer = build_vectorizer(
            FeatureConfig(name="wc", word_ngram_max=2, use_char_ngrams=True)
        )
        assert isinstance(vectorizer, FeatureUnion)

    def test_shared_preprocessor_is_embedded(self) -> None:
        """This is what keeps training and serving from drifting apart."""
        vectorizer = build_vectorizer(FeatureConfig(name="w"))
        assert vectorizer.preprocessor is feature_normalize

    def test_token_pattern_keeps_symbol_bearing_tokens(self) -> None:
        """scikit-learn's default pattern silently discards c++, c# and node.js."""
        vectorizer = build_vectorizer(FeatureConfig(name="w", min_df=1, max_df=1.0))
        vectorizer.fit(["Proficient in C++, C# and Node.js", "Python and SQL"])
        vocabulary = set(vectorizer.vocabulary_)
        assert {"c++", "c#", "node.js"} <= vocabulary

    def test_fit_transform_produces_a_matrix(self) -> None:
        vectorizer = build_vectorizer(FeatureConfig(name="w", min_df=1, max_df=1.0))
        matrix = vectorizer.fit_transform(["python docker", "chef kitchen menu"])
        assert matrix.shape[0] == 2
        assert matrix.shape[1] > 0

    def test_describe_is_serialisable(self) -> None:
        described = FeatureConfig(name="w").describe()
        assert described["name"] == "w"
        assert isinstance(described["char_ngram_range"], list)

    def test_shipped_configs_have_unique_names(self) -> None:
        names = [config.name for config in FEATURE_CONFIGS]
        assert len(names) == len(set(names))


# ===========================================================================
# Metrics
# ===========================================================================
class TestEvaluatePredictions:
    def test_perfect_predictions(self) -> None:
        truth = ["accountant", "chef", "engineering"]
        metrics = evaluate_predictions(truth, truth, LABELS)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_completely_wrong_predictions(self) -> None:
        metrics = evaluate_predictions(["accountant", "chef"], ["chef", "accountant"], LABELS)
        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0

    def test_per_class_support_is_reported(self) -> None:
        truth = ["accountant", "accountant", "chef"]
        metrics = evaluate_predictions(truth, truth, LABELS)
        assert metrics["per_class"]["accountant"]["support"] == 2
        assert metrics["per_class"]["chef"]["support"] == 1

    def test_macro_f1_punishes_a_failed_minority_class(self) -> None:
        """The reason macro-F1 is the headline metric rather than accuracy."""
        truth = ["accountant"] * 9 + ["chef"]
        predicted = ["accountant"] * 10
        metrics = evaluate_predictions(truth, predicted, LABELS)
        assert metrics["accuracy"] == 0.9
        assert metrics["macro_f1"] < 0.4

    def test_worst_class_is_surfaced(self) -> None:
        truth = ["accountant"] * 9 + ["chef"]
        predicted = ["accountant"] * 10
        metrics = evaluate_predictions(truth, predicted, LABELS)
        assert metrics["worst_class_f1"] == 0.0

    def test_unseen_label_does_not_raise(self) -> None:
        metrics = evaluate_predictions(["accountant"], ["accountant"], LABELS)
        assert metrics["per_class"]["engineering"]["support"] == 0


class TestConfusionReporting:
    def test_confusion_matrix_shape(self) -> None:
        matrix = confusion_matrix_dict(["accountant"], ["chef"], LABELS)
        assert matrix["accountant"]["chef"] == 1
        assert set(matrix) == set(LABELS)

    def test_top_confusions_excludes_the_diagonal(self) -> None:
        truth = ["accountant", "accountant", "chef"]
        predicted = ["chef", "chef", "chef"]
        confusions = top_confusions(truth, predicted, LABELS)
        assert confusions[0] == {"true": "accountant", "predicted": "chef", "count": 2}
        assert all(c["true"] != c["predicted"] for c in confusions)

    def test_top_confusions_is_deterministic(self) -> None:
        truth = ["accountant", "chef"]
        predicted = ["chef", "accountant"]
        first = top_confusions(truth, predicted, LABELS)
        second = top_confusions(truth, predicted, LABELS)
        assert first == second

    def test_top_confusions_respects_the_limit(self) -> None:
        truth = ["accountant", "chef", "engineering"]
        predicted = ["chef", "engineering", "accountant"]
        assert len(top_confusions(truth, predicted, LABELS, limit=2)) == 2

    def test_classification_report_is_text(self) -> None:
        report = text_classification_report(["chef"], ["chef"], LABELS)
        assert "chef" in report


class TestCoverageCurve:
    def test_higher_threshold_reduces_coverage(self) -> None:
        truth = ["accountant", "chef"]
        predicted = ["accountant", "accountant"]
        curve = coverage_accuracy_curve(truth, predicted, [0.9, 0.3], [0.0, 0.5])
        assert curve[0]["coverage"] == 1.0
        assert curve[1]["coverage"] == 0.5

    def test_abstaining_on_the_wrong_answer_raises_accuracy(self) -> None:
        """The trade the Uncertain verdict buys: fewer answers, more right."""
        truth = ["accountant", "chef"]
        predicted = ["accountant", "accountant"]
        curve = coverage_accuracy_curve(truth, predicted, [0.9, 0.3], [0.0, 0.5])
        assert curve[0]["accuracy_on_answered"] == 0.5
        assert curve[1]["accuracy_on_answered"] == 1.0

    def test_threshold_above_every_confidence_answers_nothing(self) -> None:
        curve = coverage_accuracy_curve(["chef"], ["chef"], [0.4], [0.9])
        assert curve[0]["coverage"] == 0.0
        assert curve[0]["accuracy_on_answered"] == 0.0


@pytest.mark.parametrize("config", FEATURE_CONFIGS, ids=lambda c: c.name)
def test_every_shipped_config_fits(config: FeatureConfig) -> None:
    """Each configuration in the search space must actually build and fit."""
    documents = [
        "senior accountant general ledger reconciliations tax",
        "executive chef kitchen menu recipes sanitation",
        "mechanical engineer autocad manufacturing tolerance",
        "accountant payroll audit budgeting forecasting",
    ]
    vectorizer = build_vectorizer(
        FeatureConfig(
            name=config.name,
            word_ngram_max=config.word_ngram_max,
            min_df=1,
            max_df=1.0,
            use_char_ngrams=config.use_char_ngrams,
        )
    )
    matrix = vectorizer.fit_transform(documents)
    assert matrix.shape[0] == len(documents)
