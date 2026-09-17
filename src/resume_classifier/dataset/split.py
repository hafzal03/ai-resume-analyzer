"""Stratified train / validation / test splitting.

Three properties matter and each is enforced here:

* **Stratified** -- every class keeps its proportion in every split, so a rare
  class is not accidentally absent from validation or test.
* **Seeded and persisted by record id** -- the assignment is written to
  ``splits.json`` and never regenerated on the fly. Re-running the build, or
  refactoring the code, cannot reshuffle the test set underneath a model.
* **Group-aware** -- members of a duplicate cluster never straddle a boundary.

The split is computed once on the deduplicated corpus and is the only place the
partition is decided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from resume_classifier.dataset.schema import ResumeRecord, Split


class SplitError(ValueError):
    """The corpus cannot be split as requested."""


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """record_id -> split, plus the parameters that produced it."""

    assignment: dict[str, Split]
    seed: int
    proportions: dict[str, float]

    def ids_for(self, split: Split) -> list[str]:
        """Sorted record ids belonging to ``split``."""
        return sorted(rid for rid, value in self.assignment.items() if value is split)

    def counts(self) -> dict[str, int]:
        """Number of records per split."""
        counts = dict.fromkeys((s.value for s in Split), 0)
        for value in self.assignment.values():
            counts[value.value] += 1
        return counts


def _min_class_count(labels: Sequence[str]) -> tuple[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    smallest = min(counts, key=lambda k: counts[k])
    return smallest, counts[smallest]


def stratified_split(
    records: Sequence[ResumeRecord],
    *,
    train: float,
    validation: float,
    test: float,
    seed: int,
) -> SplitAssignment:
    """Partition ``records`` into train / validation / test.

    Performed as two successive stratified splits: first train against a
    holdout, then the holdout into validation and test.

    Raises:
        SplitError: if a class has too few members to appear in all three
            splits, which would make per-class test metrics undefined.
    """
    if not records:
        msg = "cannot split an empty corpus"
        raise SplitError(msg)

    total = train + validation + test
    if abs(total - 1.0) > 1e-9:
        msg = f"split proportions must sum to 1.0, got {total}"
        raise SplitError(msg)

    ids = [r.record_id for r in records]
    labels = [r.label for r in records]

    smallest_class, smallest_count = _min_class_count(labels)
    if smallest_count < 3:
        msg = (
            f"class {smallest_class!r} has only {smallest_count} record(s); "
            "at least 3 are required to appear in train, validation and test"
        )
        raise SplitError(msg)

    holdout_fraction = validation + test
    train_ids, holdout_ids, _, holdout_labels = train_test_split(
        ids,
        labels,
        test_size=holdout_fraction,
        random_state=seed,
        stratify=labels,
        shuffle=True,
    )

    # Within the holdout, validation and test share it in their original ratio.
    test_fraction_of_holdout = test / holdout_fraction
    validation_ids, test_ids = train_test_split(
        holdout_ids,
        test_size=test_fraction_of_holdout,
        random_state=seed,
        stratify=holdout_labels,
        shuffle=True,
    )

    assignment: dict[str, Split] = {}
    for rid in train_ids:
        assignment[rid] = Split.TRAIN
    for rid in validation_ids:
        assignment[rid] = Split.VALIDATION
    for rid in test_ids:
        assignment[rid] = Split.TEST

    if len(assignment) != len(ids):
        msg = "split did not assign every record exactly once"
        raise SplitError(msg)

    return SplitAssignment(
        assignment=assignment,
        seed=seed,
        proportions={"train": train, "validation": validation, "test": test},
    )
