"""Confidence policy: when to answer, and when to say "Uncertain".

Two signals are combined, because they fail in different ways:

* **Top probability** -- how sure the model is about its best guess.
* **Margin** -- how far ahead that guess is of the runner-up. A model can be
  0.45 confident and still decisive (runner-up at 0.05), or 0.45 confident and
  genuinely torn (runner-up at 0.43). Probability alone cannot tell those apart.

Both thresholds are tuned on the **validation** split. Tuning them on test would
turn the held-out estimate into a fitted one.

The policy is chosen to maximise accuracy on the answered subset while keeping
coverage above a floor -- abstaining on everything would be perfectly accurate
and perfectly useless.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Never abstain on more than this fraction of inputs.
MIN_COVERAGE = 0.80

_THRESHOLD_GRID = (0.0, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70)
_MARGIN_GRID = (0.0, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Thresholds deciding whether a prediction is committed to."""

    threshold: float
    margin: float

    def is_confident(self, top_probability: float, runner_up: float) -> bool:
        """Whether a prediction clears both thresholds."""
        return top_probability >= self.threshold and (top_probability - runner_up) >= self.margin


@dataclass(frozen=True, slots=True)
class Verdict:
    """The outcome of applying a policy to one probability distribution."""

    label: str | None
    confidence: float
    margin: float
    is_uncertain: bool
    ranked: list[tuple[str, float]]

    @property
    def display_label(self) -> str:
        """Label to show a user, or ``"Uncertain"`` when abstaining."""
        return "Uncertain" if self.is_uncertain or self.label is None else self.label


def rank_probabilities(
    probabilities: Sequence[float], classes: Sequence[str], limit: int = 5
) -> list[tuple[str, float]]:
    """Classes ordered by probability, highest first."""
    order = np.argsort(np.asarray(probabilities))[::-1][:limit]
    return [(classes[int(i)], round(float(probabilities[int(i)]), 4)) for i in order]


def apply_policy(
    probabilities: Sequence[float],
    classes: Sequence[str],
    policy: ConfidencePolicy,
) -> Verdict:
    """Turn one probability row into a :class:`Verdict`."""
    array = np.asarray(probabilities, dtype=float)
    if array.size == 0:
        return Verdict(None, 0.0, 0.0, True, [])

    ordered = np.sort(array)[::-1]
    top = float(ordered[0])
    runner_up = float(ordered[1]) if ordered.size > 1 else 0.0
    best_label = classes[int(np.argmax(array))]
    confident = policy.is_confident(top, runner_up)

    return Verdict(
        label=best_label,
        confidence=round(top, 4),
        margin=round(top - runner_up, 4),
        is_uncertain=not confident,
        ranked=rank_probabilities([float(v) for v in array], classes),
    )


def tune_policy(
    *,
    y_true: Sequence[str],
    probabilities: Any,
    classes: Sequence[str],
) -> tuple[ConfidencePolicy, dict[str, Any]]:
    """Search the threshold grid for the best policy on validation data.

    Returns the chosen policy plus a report describing the trade-off, which is
    stored in model metadata so the decision stays inspectable.
    """
    proba = np.asarray(probabilities, dtype=float)
    truth = np.asarray(y_true)
    class_array = np.asarray(classes)

    predicted = class_array[np.argmax(proba, axis=1)]
    correct = predicted == truth

    ordered = np.sort(proba, axis=1)[:, ::-1]
    top = ordered[:, 0]
    runner_up = ordered[:, 1] if ordered.shape[1] > 1 else np.zeros_like(top)
    margins = top - runner_up

    total = len(truth)
    baseline_accuracy = float(correct.mean()) if total else 0.0

    evaluated: list[dict[str, float]] = []
    best: tuple[float, float, float, float] | None = None  # (acc, coverage, thr, margin)

    for threshold in _THRESHOLD_GRID:
        for margin in _MARGIN_GRID:
            answered_mask = (top >= threshold) & (margins >= margin)
            answered = int(answered_mask.sum())
            coverage = answered / total if total else 0.0
            accuracy = float(correct[answered_mask].mean()) if answered else 0.0
            evaluated.append(
                {
                    "threshold": threshold,
                    "margin": margin,
                    "coverage": round(coverage, 4),
                    "accuracy_on_answered": round(accuracy, 4),
                }
            )
            if coverage < MIN_COVERAGE:
                continue
            key = (accuracy, coverage, -threshold, -margin)
            if best is None or key > (best[0], best[1], -best[2], -best[3]):
                best = (accuracy, coverage, threshold, margin)

    if best is None:
        # Nothing met the coverage floor: answer everything rather than abstain
        # on inputs the system is meant to serve.
        policy = ConfidencePolicy(threshold=0.0, margin=0.0)
        report: dict[str, Any] = {
            "selected": {"threshold": 0.0, "margin": 0.0},
            "note": "no policy met the coverage floor; abstention disabled",
            "min_coverage": MIN_COVERAGE,
            "baseline_accuracy_all": round(baseline_accuracy, 4),
            "grid": evaluated,
        }
        return policy, report

    accuracy, coverage, threshold, margin = best
    policy = ConfidencePolicy(threshold=float(threshold), margin=float(margin))
    report = {
        "selected": {
            "threshold": float(threshold),
            "margin": float(margin),
            "coverage": round(coverage, 4),
            "accuracy_on_answered": round(accuracy, 4),
        },
        "min_coverage": MIN_COVERAGE,
        "baseline_accuracy_all": round(baseline_accuracy, 4),
        "accuracy_gain": round(accuracy - baseline_accuracy, 4),
        "grid": evaluated,
    }
    return policy, report
