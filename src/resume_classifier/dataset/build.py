"""The dataset build pipeline.

One command turns an immutable raw corpus into a validated, deduplicated,
split corpus with a full provenance trail:

    raw rows -> validate -> deduplicate -> split -> corpus.jsonl + splits.json

Every rejected row is written to ``quarantine.jsonl`` with a reason code, and
the build asserts ``total_in == kept + quarantined`` before it finishes. Nothing
disappears silently.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from resume_classifier.config import Settings
from resume_classifier.dataset.dedup import find_duplicates
from resume_classifier.dataset.ingest import get_adapter
from resume_classifier.dataset.manifest import (
    git_commit,
    sha256_file,
    utc_now,
    write_manifest,
)
from resume_classifier.dataset.report import build_report, render_report_markdown
from resume_classifier.dataset.schema import (
    RejectedRecord,
    RejectReason,
    ResumeRecord,
    Split,
    content_hash,
    make_record_id,
)
from resume_classifier.dataset.split import stratified_split
from resume_classifier.dataset.taxonomy import load_taxonomy
from resume_classifier.preprocessing.text import (
    feature_normalize,
    ingestion_normalize,
    word_count,
)

logger = logging.getLogger(__name__)

CORPUS_FILENAME = "corpus.jsonl"
SPLITS_FILENAME = "splits.json"
QUARANTINE_FILENAME = "quarantine.jsonl"
REPORT_JSON_FILENAME = "report.json"
REPORT_MD_FILENAME = "report.md"
MANIFEST_FILENAME = "MANIFEST.json"


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Summary of a completed dataset build."""

    output_dir: Path
    total_in: int
    kept: int
    quarantined: int
    split_counts: dict[str, int]
    class_count: int

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{self.kept} records kept, {self.quarantined} quarantined "
            f"({self.total_in} read) across {self.class_count} classes -> "
            f"{self.output_dir}"
        )


def build_dataset(settings: Settings, *, source: str = "livecareer_resumes") -> BuildResult:
    """Build the processed corpus from an immutable raw source.

    Args:
        settings: Configuration supplying paths, seed, thresholds.
        source: Registered source adapter name.

    Returns:
        A :class:`BuildResult` describing what was produced.
    """
    taxonomy_path = settings.taxonomy_dir / "categories_v1.toml"
    taxonomy = load_taxonomy(taxonomy_path)
    adapter = get_adapter(source, settings.raw_data_dir)

    logger.info("reading corpus source=%s version=%s", adapter.source, adapter.source_version)

    ingested_at = utc_now()
    candidates: list[ResumeRecord] = []
    rejected: list[RejectedRecord] = []
    seen_ids: set[str] = set()
    total_in = 0
    unmapped_labels: Counter[str] = Counter()

    # -- validation gate --------------------------------------------------
    for row in adapter.rows():
        total_in += 1
        text = ingestion_normalize(row.text)
        chars = len(text)

        if not row.label_raw.strip():
            rejected.append(
                RejectedRecord(
                    source=adapter.source,
                    source_id=row.source_id,
                    reason=RejectReason.LABEL_MISSING,
                    text_length_chars=chars,
                )
            )
            continue

        label = taxonomy.map_label(row.label_raw)
        if label is None:
            unmapped_labels[row.label_raw] += 1
            rejected.append(
                RejectedRecord(
                    source=adapter.source,
                    source_id=row.source_id,
                    reason=RejectReason.LABEL_UNMAPPED,
                    detail=f"no taxonomy class for {row.label_raw!r}",
                    label_raw=row.label_raw,
                    text_length_chars=chars,
                )
            )
            continue

        if not text:
            rejected.append(
                RejectedRecord(
                    source=adapter.source,
                    source_id=row.source_id,
                    reason=RejectReason.TEXT_EMPTY,
                    label_raw=row.label_raw,
                    text_length_chars=0,
                )
            )
            continue

        words = word_count(feature_normalize(text))
        if chars < settings.min_resume_chars or words < settings.min_resume_words:
            rejected.append(
                RejectedRecord(
                    source=adapter.source,
                    source_id=row.source_id,
                    reason=RejectReason.TEXT_TOO_SHORT,
                    detail=f"{chars} chars, {words} words",
                    label_raw=row.label_raw,
                    text_length_chars=chars,
                )
            )
            continue

        record_id = make_record_id(adapter.source, row.source_id)
        if record_id in seen_ids:
            rejected.append(
                RejectedRecord(
                    source=adapter.source,
                    source_id=row.source_id,
                    reason=RejectReason.ID_DUPLICATE,
                    detail=f"record_id {record_id} already seen",
                    label_raw=row.label_raw,
                    text_length_chars=chars,
                )
            )
            continue
        seen_ids.add(record_id)

        candidates.append(
            ResumeRecord(
                record_id=record_id,
                source=adapter.source,
                source_version=adapter.source_version,
                source_id=row.source_id,
                resume_text=text,
                label=label,
                label_raw=row.label_raw,
                text_sha256=content_hash(text),
                text_length_chars=chars,
                text_length_words=words,
                ingested_at=ingested_at,
            )
        )

    if unmapped_labels:
        logger.warning(
            "quarantined %d row(s) with unmapped labels: %s",
            sum(unmapped_labels.values()),
            dict(unmapped_labels),
        )

    logger.info("validated %d/%d rows", len(candidates), total_in)

    # -- deduplication (before splitting) ---------------------------------
    clusters = find_duplicates(
        [feature_normalize(r.resume_text) for r in candidates],
        [r.text_sha256 for r in candidates],
        threshold=settings.near_duplicate_threshold,
    )
    drop = clusters.duplicate_indices
    logger.info(
        "duplicate detection: %d cluster(s) with more than one member, %d record(s) collapsed",
        clusters.cluster_count,
        len(drop),
    )

    kept: list[ResumeRecord] = []
    for index, record in enumerate(candidates):
        group = clusters.group_of[index]
        members = clusters.members[group]
        if index in drop:
            representative = candidates[clusters.representative_of[group]]
            reason = (
                RejectReason.EXACT_DUPLICATE
                if record.text_sha256 == representative.text_sha256
                else RejectReason.NEAR_DUPLICATE
            )
            rejected.append(
                RejectedRecord(
                    source=record.source,
                    source_id=record.source_id,
                    reason=reason,
                    detail=f"cluster {group}, representative {representative.record_id}",
                    label_raw=record.label_raw,
                    text_length_chars=record.text_length_chars,
                )
            )
            continue
        kept.append(
            record.model_copy(update={"duplicate_group_id": group if len(members) > 1 else None})
        )

    if not kept:
        msg = "validation and deduplication removed every record"
        raise ValueError(msg)

    # -- split -------------------------------------------------------------
    assignment = stratified_split(
        kept,
        train=settings.split_train,
        validation=settings.split_validation,
        test=settings.split_test,
        seed=settings.random_seed,
    )

    # -- reconcile ---------------------------------------------------------
    if len(kept) + len(rejected) != total_in:
        msg = (
            f"reconciliation failed: read {total_in}, kept {len(kept)}, quarantined {len(rejected)}"
        )
        raise ValueError(msg)

    # -- write outputs ------------------------------------------------------
    out = settings.dataset_dir
    out.mkdir(parents=True, exist_ok=True)

    kept_sorted = sorted(kept, key=lambda r: r.record_id)
    _write_jsonl(out / CORPUS_FILENAME, (r.model_dump() for r in kept_sorted))
    _write_jsonl(
        out / QUARANTINE_FILENAME,
        (r.model_dump() for r in sorted(rejected, key=lambda r: (r.reason.value, r.source_id))),
    )

    (out / SPLITS_FILENAME).write_text(
        json.dumps(
            {
                "seed": assignment.seed,
                "proportions": assignment.proportions,
                "counts": assignment.counts(),
                "assignment": {
                    rid: assignment.assignment[rid].value for rid in sorted(assignment.assignment)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report = build_report(
        records=kept_sorted,
        rejected=rejected,
        assignment=assignment,
        taxonomy=taxonomy,
        total_in=total_in,
        duplicate_clusters=clusters.cluster_count,
    )
    (out / REPORT_JSON_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out / REPORT_MD_FILENAME).write_text(
        render_report_markdown(report, taxonomy), encoding="utf-8"
    )

    raw_dir = settings.raw_data_dir / source / adapter.source_version
    write_manifest(
        out / MANIFEST_FILENAME,
        {
            "dataset_version": settings.dataset_version,
            "built_at": ingested_at,
            "git_commit": git_commit(settings.project_root),
            "source": {
                "name": adapter.source,
                "version": adapter.source_version,
                "files": [
                    {"file": p.name, "sha256": sha256_file(p)}
                    for p in sorted(raw_dir.glob("*"))
                    if p.is_file()
                ],
            },
            "taxonomy": {
                "path": str(taxonomy_path.relative_to(settings.project_root)),
                "version": taxonomy.version,
                "class_count": len(taxonomy.classes),
            },
            "parameters": {
                "random_seed": settings.random_seed,
                "min_resume_chars": settings.min_resume_chars,
                "min_resume_words": settings.min_resume_words,
                "near_duplicate_threshold": settings.near_duplicate_threshold,
                "split_train": settings.split_train,
                "split_validation": settings.split_validation,
                "split_test": settings.split_test,
            },
            "counts": {
                "read": total_in,
                "kept": len(kept),
                "quarantined": len(rejected),
                **assignment.counts(),
            },
            "outputs": {
                name: sha256_file(out / name)
                for name in (CORPUS_FILENAME, SPLITS_FILENAME, QUARANTINE_FILENAME)
            },
        },
    )

    result = BuildResult(
        output_dir=out,
        total_in=total_in,
        kept=len(kept),
        quarantined=len(rejected),
        split_counts=assignment.counts(),
        class_count=len({r.label for r in kept}),
    )
    logger.info("dataset build complete: %s", result.summary())
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dicts as newline-delimited JSON.

    Keys are sorted and the newline is forced to ``\\n`` so that rebuilding the
    same corpus produces a byte-identical file on every platform.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_corpus(dataset_dir: Path) -> list[ResumeRecord]:
    """Read ``corpus.jsonl`` back into validated records."""
    path = dataset_dir / CORPUS_FILENAME
    if not path.is_file():
        msg = f"corpus not found at {path}; run 'rc-build-dataset' first"
        raise FileNotFoundError(msg)
    with path.open("r", encoding="utf-8") as handle:
        return [ResumeRecord.model_validate_json(line) for line in handle if line.strip()]


def load_splits(dataset_dir: Path) -> dict[str, Split]:
    """Read ``splits.json`` back into a record_id -> split mapping."""
    path = dataset_dir / SPLITS_FILENAME
    if not path.is_file():
        msg = f"splits not found at {path}; run 'rc-build-dataset' first"
        raise FileNotFoundError(msg)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {rid: Split(value) for rid, value in data["assignment"].items()}
