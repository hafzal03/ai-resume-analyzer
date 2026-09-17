"""Dataset profiling and quality reporting.

The report is produced by the build itself rather than by an ad-hoc notebook, so
every rebuild carries current numbers. It is written twice: JSON for machines
and tests, Markdown for humans.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from resume_classifier.dataset.schema import RejectedRecord, ResumeRecord, Split
from resume_classifier.dataset.split import SplitAssignment
from resume_classifier.dataset.taxonomy import CategoryTaxonomy


def _percentiles(values: Sequence[int]) -> dict[str, float]:
    """Summary statistics for a numeric column."""
    if not values:
        return {}
    ordered = sorted(values)

    def pct(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p05": pct(0.05),
        "p25": pct(0.25),
        "median": float(statistics.median(ordered)),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "max": float(ordered[-1]),
        "mean": round(statistics.fmean(ordered), 2),
    }


def build_report(
    *,
    records: Sequence[ResumeRecord],
    rejected: Sequence[RejectedRecord],
    assignment: SplitAssignment,
    taxonomy: CategoryTaxonomy,
    total_in: int,
    duplicate_clusters: int,
) -> dict[str, Any]:
    """Assemble the dataset quality report as a plain dictionary."""
    label_counts = Counter(r.label for r in records)
    reject_counts = Counter(r.reason.value for r in rejected)

    per_split_labels: dict[str, Counter[str]] = {s.value: Counter() for s in Split}
    for record in records:
        split = assignment.assignment[record.record_id]
        per_split_labels[split.value][record.label] += 1

    smallest = min(label_counts.values()) if label_counts else 0
    largest = max(label_counts.values()) if label_counts else 0

    return {
        "totals": {
            "read": total_in,
            "kept": len(records),
            "quarantined": len(rejected),
            "classes": len(label_counts),
            "duplicate_clusters": duplicate_clusters,
        },
        "reconciliation_ok": len(records) + len(rejected) == total_in,
        "splits": assignment.counts(),
        "split_proportions": assignment.proportions,
        "seed": assignment.seed,
        "class_distribution": dict(sorted(label_counts.items())),
        "class_balance": {
            "smallest_class_count": smallest,
            "largest_class_count": largest,
            "imbalance_ratio": round(largest / smallest, 2) if smallest else None,
        },
        "class_distribution_by_split": {
            split: dict(sorted(counts.items())) for split, counts in per_split_labels.items()
        },
        "text_length_chars": _percentiles([r.text_length_chars for r in records]),
        "text_length_words": _percentiles([r.text_length_words for r in records]),
        "quarantine_reasons": dict(sorted(reject_counts.items())),
        "taxonomy_version": taxonomy.version,
    }


def render_report_markdown(report: dict[str, Any], taxonomy: CategoryTaxonomy) -> str:
    """Render the report as Markdown for human review."""
    totals = report["totals"]
    lines: list[str] = [
        "# Dataset report",
        "",
        f"- Rows read: **{totals['read']}**",
        f"- Records kept: **{totals['kept']}**",
        f"- Quarantined: **{totals['quarantined']}**",
        f"- Classes: **{totals['classes']}** (taxonomy v{report['taxonomy_version']})",
        f"- Duplicate clusters collapsed: **{totals['duplicate_clusters']}**",
        f"- Reconciliation (`read == kept + quarantined`): "
        f"**{'OK' if report['reconciliation_ok'] else 'FAILED'}**",
        f"- Split seed: `{report['seed']}`",
        "",
        "## Splits",
        "",
        "| Split | Records |",
        "| --- | ---: |",
    ]
    for split, count in report["splits"].items():
        lines.append(f"| {split} | {count} |")

    balance = report["class_balance"]
    lines += [
        "",
        "## Class distribution",
        "",
        f"Smallest class: {balance['smallest_class_count']} · "
        f"largest: {balance['largest_class_count']} · "
        f"imbalance ratio: {balance['imbalance_ratio']}x",
        "",
        "| Class | Total | Train | Validation | Test |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_split = report["class_distribution_by_split"]
    for class_id, total in report["class_distribution"].items():
        lines.append(
            f"| {taxonomy.describe(class_id)} | {total} "
            f"| {by_split['train'].get(class_id, 0)} "
            f"| {by_split['validation'].get(class_id, 0)} "
            f"| {by_split['test'].get(class_id, 0)} |"
        )

    lines += ["", "## Text length", "", "| Metric | Characters | Words |", "| --- | ---: | ---: |"]
    chars, words = report["text_length_chars"], report["text_length_words"]
    for key in ("min", "p05", "p25", "median", "p75", "p95", "max", "mean"):
        lines.append(f"| {key} | {chars.get(key, 0):,.0f} | {words.get(key, 0):,.0f} |")

    if report["quarantine_reasons"]:
        lines += ["", "## Quarantine reasons", "", "| Reason | Count |", "| --- | ---: |"]
        for reason, count in report["quarantine_reasons"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines += ["", "## Quarantine reasons", "", "No records were rejected."]

    return "\n".join(lines) + "\n"
