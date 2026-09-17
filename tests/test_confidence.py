"""Tests for the confidence and abstention policy.

The behaviour being protected: the system says "Uncertain" when the evidence is
weak, instead of forcing every document into a category.
"""

from __future__ import annotations

import numpy as np

from resume_classifier.inference.confidence import (
    MIN_COVERAGE,
    ConfidencePolicy,
    apply_policy,
    rank_probabilities,
    tune_policy,
)

CLASSES = ["accountant", "chef", "engineering"]


class TestPolicy:
    def test_confident_prediction_is_committed_to(self) -> None:
        policy = ConfidencePolicy(threshold=0.5, margin=0.1)
        verdict = apply_policy([0.8, 0.15, 0.05], CLASSES, policy)
        assert verdict.label == "accountant"
        assert not verdict.is_uncertain
        assert verdict.display_label == "accountant"

    def test_low_probability_triggers_abstention(self) -> None:
        policy = ConfidencePolicy(threshold=0.6, margin=0.0)
        verdict = apply_policy([0.4, 0.35, 0.25], CLASSES, policy)
        assert verdict.is_uncertain
        assert verdict.display_label == "Uncertain"

    def test_narrow_margin_triggers_abstention(self) -> None:
        """A 0.45/0.44 split is a coin flip even though 0.45 clears a threshold."""
        policy = ConfidencePolicy(threshold=0.4, margin=0.10)
        verdict = apply_policy([0.45, 0.44, 0.11], CLASSES, policy)
        assert verdict.is_uncertain

    def test_wide_margin_at_the_same_probability_is_accepted(self) -> None:
        policy = ConfidencePolicy(threshold=0.4, margin=0.10)
        verdict = apply_policy([0.45, 0.05, 0.50], CLASSES, policy)
        assert verdict.label == "engineering"

    def test_best_label_is_still_reported_when_uncertain(self) -> None:
        """Abstaining publicly should not discard the ranking underneath."""
        policy = ConfidencePolicy(threshold=0.9, margin=0.0)
        verdict = apply_policy([0.5, 0.3, 0.2], CLASSES, policy)
        assert verdict.is_uncertain
        assert verdict.label == "accountant"
        assert verdict.ranked[0][0] == "accountant"

    def test_empty_distribution_is_uncertain(self) -> None:
        verdict = apply_policy([], [], ConfidencePolicy(0.5, 0.1))
        assert verdict.is_uncertain
        assert verdict.label is None


def test_rank_probabilities_is_descending() -> None:
    ranked = rank_probabilities([0.2, 0.5, 0.3], CLASSES)
    assert [label for label, _ in ranked] == ["chef", "engineering", "accountant"]


class TestTuning:
    def test_tuning_respects_the_coverage_floor(self) -> None:
        """Abstaining on everything would be perfectly accurate and useless."""
        rng = np.random.default_rng(0)
        probabilities = rng.dirichlet(np.ones(3), size=300)
        truth = [CLASSES[int(np.argmax(row))] for row in probabilities]
        _, report = tune_policy(y_true=truth, probabilities=probabilities, classes=CLASSES)
        selected = report.get("selected", {})
        if "coverage" in selected:
            assert selected["coverage"] >= MIN_COVERAGE

    def test_tuning_prefers_accuracy_on_answered(self) -> None:
        # Confident rows are correct; unconfident rows are wrong.
        confident = np.tile([0.9, 0.05, 0.05], (80, 1))
        unsure = np.tile([0.4, 0.35, 0.25], (20, 1))
        probabilities = np.vstack([confident, unsure])
        truth = ["accountant"] * 80 + ["chef"] * 20

        policy, report = tune_policy(y_true=truth, probabilities=probabilities, classes=CLASSES)
        assert report["selected"]["accuracy_on_answered"] >= report["baseline_accuracy_all"]
        assert policy.threshold >= 0.0

    def test_report_records_the_whole_grid(self) -> None:
        """The abstention decision must stay inspectable after the fact."""
        probabilities = np.tile([0.7, 0.2, 0.1], (50, 1))
        truth = ["accountant"] * 50
        _, report = tune_policy(y_true=truth, probabilities=probabilities, classes=CLASSES)
        assert report["grid"]
        assert {"threshold", "margin", "coverage"} <= set(report["grid"][0])
