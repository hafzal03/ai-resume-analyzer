"""Model evaluation.

Macro-F1 is the headline metric, not accuracy. With a 5x class imbalance, a
model can post a respectable accuracy while failing a minority class entirely;
macro-F1 makes that visible because every class contributes equally.

Accuracy is still reported, because it is what people ask about -- but it is
never the selection criterion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Compute the full metric set for one set of predictions."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    # `labels` must be passed explicitly. Without it, scikit-learn averages over
    # only the classes observed in this particular sample, so a class that never
    # appears silently drops out of the denominator -- making macro-F1 look
    # better and, worse, not comparable between runs.
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
            4,
        ),
        "weighted_f1": round(
            float(
                f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
            ),
            4,
        ),
        "per_class": per_class,
        "worst_class": min(per_class, key=lambda k: per_class[k]["f1"]),
        "worst_class_f1": round(min(v["f1"] for v in per_class.values()), 4),
    }


def confusion_matrix_dict(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Confusion matrix as nested dicts (true label -> predicted label -> count)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        true_label: {pred_label: int(matrix[i][j]) for j, pred_label in enumerate(labels)}
        for i, true_label in enumerate(labels)
    }


def text_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> str:
    """scikit-learn's human-readable per-class report."""
    report: str = classification_report(
        y_true, y_pred, labels=list(labels), zero_division=0, digits=3
    )
    return report


def top_confusions(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """The most frequent off-diagonal confusions, largest first.

    More useful than the full matrix when reviewing a 24-class model: it answers
    "what does this model actually get wrong" in one glance.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    pairs: list[tuple[int, str, str]] = [
        (int(matrix[i][j]), labels[i], labels[j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and matrix[i][j] > 0
    ]
    # Largest first, then alphabetically, so the output is deterministic.
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    return [
        {"true": true_label, "predicted": predicted, "count": count}
        for count, true_label, predicted in pairs[:limit]
    ]


def coverage_accuracy_curve(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    confidences: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, float]]:
    """Accuracy against coverage as the abstention threshold varies.

    This is the trade-off an ``Uncertain`` verdict buys: answering fewer
    questions in exchange for being right more often on the ones answered.
    """
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)
    conf_arr = np.asarray(confidences, dtype=float)

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        mask = conf_arr >= threshold
        answered = int(mask.sum())
        coverage = answered / len(true_arr) if len(true_arr) else 0.0
        accuracy = float((true_arr[mask] == pred_arr[mask]).mean()) if answered else 0.0
        curve.append(
            {
                "threshold": round(float(threshold), 3),
                "coverage": round(coverage, 4),
                "answered": float(answered),
                "accuracy_on_answered": round(accuracy, 4),
            }
        )
    return curve
