"""Tests for dataset engineering.

The leakage assertions here are the important ones. They encode the properties
that make the reported metrics trustworthy, so that a future refactor cannot
quietly reintroduce train/test contamination.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resume_classifier.config import Settings
from resume_classifier.dataset.dedup import find_duplicates
from resume_classifier.dataset.schema import (
    RejectReason,
    ResumeRecord,
    Split,
    content_hash,
    make_record_id,
)
from resume_classifier.dataset.split import SplitError, stratified_split
from resume_classifier.dataset.taxonomy import TaxonomyError, load_taxonomy
from tests.conftest import requires_corpus


# ===========================================================================
# Schema
# ===========================================================================
class TestSchema:
    def test_record_id_is_deterministic(self) -> None:
        assert make_record_id("src", "42") == make_record_id("src", "42")

    def test_record_id_depends_on_source(self) -> None:
        assert make_record_id("a", "42") != make_record_id("b", "42")

    def test_content_hash_ignores_formatting(self) -> None:
        """Formatting-only differences must collapse to one hash."""
        assert content_hash("Python, Docker") == content_hash("python\ndocker")

    def test_content_hash_distinguishes_content(self) -> None:
        assert content_hash("python") != content_hash("java")


# ===========================================================================
# Category taxonomy
# ===========================================================================
class TestCategoryTaxonomy:
    def test_real_taxonomy_loads(self, taxonomy_dir: Path) -> None:
        taxonomy = load_taxonomy(taxonomy_dir / "categories_v1.toml")
        assert taxonomy.version == 1
        assert len(taxonomy.classes) == 24

    def test_mapping_is_case_insensitive(self, taxonomy_dir: Path) -> None:
        taxonomy = load_taxonomy(taxonomy_dir / "categories_v1.toml")
        assert taxonomy.map_label("information-technology") == "information_technology"
        assert taxonomy.map_label("INFORMATION-TECHNOLOGY") == "information_technology"

    def test_unknown_label_maps_to_none(self, taxonomy_dir: Path) -> None:
        """An unmapped label must be visible, not silently bucketed."""
        assert load_taxonomy(taxonomy_dir / "categories_v1.toml").map_label("XYZ") is None

    def test_duplicate_source_label_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "t.toml"
        path.write_text(
            "taxonomy_version = 1\n"
            '[[classes]]\nid = "a"\nsource_labels = ["X"]\n'
            '[[classes]]\nid = "b"\nsource_labels = ["X"]\n',
            encoding="utf-8",
        )
        with pytest.raises(TaxonomyError, match="claimed by both"):
            load_taxonomy(path)

    def test_no_catch_all_class(self, taxonomy_dir: Path) -> None:
        """A heterogeneous "other" bucket has no coherent decision boundary."""
        ids = load_taxonomy(taxonomy_dir / "categories_v1.toml").class_ids
        assert "other" not in ids
        assert "misc" not in ids


# ===========================================================================
# Deduplication
# ===========================================================================
class TestDeduplication:
    def test_exact_duplicates_are_clustered(self) -> None:
        texts = ["alpha beta", "alpha beta", "gamma delta"]
        hashes = [content_hash(t) for t in texts]
        clusters = find_duplicates(texts, hashes, threshold=0.99)
        assert len(clusters.duplicate_indices) == 1

    def test_distinct_documents_are_not_clustered(self) -> None:
        texts = ["accounting ledger tax", "aircraft flight avionics"]
        hashes = [content_hash(t) for t in texts]
        clusters = find_duplicates(texts, hashes, threshold=0.95)
        assert clusters.duplicate_indices == set()

    def test_representative_is_the_lowest_index(self) -> None:
        texts = ["same text here", "same text here"]
        hashes = [content_hash(t) for t in texts]
        clusters = find_duplicates(texts, hashes)
        assert 0 in clusters.representative_of.values()
        assert clusters.duplicate_indices == {1}

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            find_duplicates(["a"], ["h1", "h2"])


# ===========================================================================
# Splitting
# ===========================================================================
def _record(index: int, label: str) -> ResumeRecord:
    return ResumeRecord(
        record_id=f"r{index:04d}",
        source="test",
        source_version="v1",
        source_id=str(index),
        resume_text="text " * 60,
        label=label,
        label_raw=label.upper(),
        text_sha256="0" * 64,
        text_length_chars=300,
        text_length_words=60,
        ingested_at="2026-01-01T00:00:00+00:00",
    )


class TestSplitting:
    def test_every_record_is_assigned_once(self) -> None:
        records = [_record(i, f"c{i % 4}") for i in range(200)]
        assignment = stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=42)
        assert len(assignment.assignment) == len(records)
        assert sum(assignment.counts().values()) == len(records)

    def test_splits_are_disjoint(self) -> None:
        records = [_record(i, f"c{i % 4}") for i in range(200)]
        assignment = stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=42)
        train = set(assignment.ids_for(Split.TRAIN))
        validation = set(assignment.ids_for(Split.VALIDATION))
        test = set(assignment.ids_for(Split.TEST))
        assert train & test == set()
        assert train & validation == set()
        assert validation & test == set()

    def test_same_seed_gives_same_split(self) -> None:
        """Splits must survive a rebuild, or every past metric is unreproducible."""
        records = [_record(i, f"c{i % 4}") for i in range(200)]
        first = stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=7)
        second = stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=7)
        assert first.assignment == second.assignment

    def test_every_class_appears_in_every_split(self) -> None:
        records = [_record(i, f"c{i % 4}") for i in range(200)]
        assignment = stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=42)
        by_id = {r.record_id: r for r in records}
        for split in Split:
            labels = {by_id[rid].label for rid in assignment.ids_for(split)}
            assert len(labels) == 4

    def test_rejects_class_too_small_to_split(self) -> None:
        records = [_record(i, "a") for i in range(30)] + [_record(99, "rare")]
        with pytest.raises(SplitError, match="at least 3"):
            stratified_split(records, train=0.7, validation=0.15, test=0.15, seed=42)

    def test_rejects_proportions_that_do_not_sum_to_one(self) -> None:
        records = [_record(i, f"c{i % 3}") for i in range(60)]
        with pytest.raises(SplitError, match=r"sum to 1\.0"):
            stratified_split(records, train=0.5, validation=0.2, test=0.2, seed=42)

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(SplitError, match="empty corpus"):
            stratified_split([], train=0.7, validation=0.15, test=0.15, seed=42)


# ===========================================================================
# Properties of the BUILT corpus -- these are the leakage guards
# ===========================================================================
class TestBuiltCorpus:
    def test_reconciliation(self, settings: Settings, dataset_dir: Path) -> None:
        """Every input row is either kept or quarantined. Nothing vanishes."""
        requires_corpus(dataset_dir)
        report = json.loads((dataset_dir / "report.json").read_text(encoding="utf-8"))
        totals = report["totals"]
        assert report["reconciliation_ok"]
        assert totals["read"] == totals["kept"] + totals["quarantined"]

    def test_no_record_id_crosses_splits(self, dataset_dir: Path) -> None:
        requires_corpus(dataset_dir)
        from resume_classifier.dataset import load_splits

        splits = load_splits(dataset_dir)
        train = {r for r, s in splits.items() if s is Split.TRAIN}
        test = {r for r, s in splits.items() if s is Split.TEST}
        validation = {r for r, s in splits.items() if s is Split.VALIDATION}
        assert train & test == set()
        assert train & validation == set()
        assert validation & test == set()

    def test_no_content_hash_crosses_splits(self, dataset_dir: Path) -> None:
        """The decisive leakage check: the same resume in train and test would
        inflate every metric the project reports."""
        requires_corpus(dataset_dir)
        from resume_classifier.dataset import load_corpus, load_splits

        records = load_corpus(dataset_dir)
        splits = load_splits(dataset_dir)
        train_hashes = {r.text_sha256 for r in records if splits[r.record_id] is Split.TRAIN}
        test_hashes = {r.text_sha256 for r in records if splits[r.record_id] is Split.TEST}
        assert train_hashes & test_hashes == set()

    def test_all_records_have_a_split(self, dataset_dir: Path) -> None:
        requires_corpus(dataset_dir)
        from resume_classifier.dataset import load_corpus, load_splits

        records = load_corpus(dataset_dir)
        splits = load_splits(dataset_dir)
        assert all(r.record_id in splits for r in records)

    def test_labels_are_all_in_the_taxonomy(self, dataset_dir: Path, taxonomy_dir: Path) -> None:
        requires_corpus(dataset_dir)
        from resume_classifier.dataset import load_corpus

        valid = set(load_taxonomy(taxonomy_dir / "categories_v1.toml").class_ids)
        assert {r.label for r in load_corpus(dataset_dir)} <= valid

    def test_quarantine_reasons_are_known_codes(self, dataset_dir: Path) -> None:
        requires_corpus(dataset_dir)
        known = {r.value for r in RejectReason}
        path = dataset_dir / "quarantine.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    assert json.loads(line)["reason"] in known

    def test_manifest_records_provenance(self, dataset_dir: Path) -> None:
        """Data is git-ignored; the manifest is what makes it reproducible."""
        requires_corpus(dataset_dir)
        manifest = json.loads((dataset_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["source"]["files"]
        assert all(f["sha256"] for f in manifest["source"]["files"])
        assert manifest["parameters"]["random_seed"] is not None
        assert manifest["outputs"]["corpus.jsonl"]
