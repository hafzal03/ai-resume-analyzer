"""Dataset engineering: ingestion, validation, deduplication, splitting."""

from resume_classifier.dataset.build import (
    BuildResult,
    build_dataset,
    load_corpus,
    load_splits,
)
from resume_classifier.dataset.schema import (
    RejectedRecord,
    RejectReason,
    ResumeRecord,
    Split,
)
from resume_classifier.dataset.taxonomy import CategoryTaxonomy, load_taxonomy

__all__ = [
    "BuildResult",
    "CategoryTaxonomy",
    "RejectReason",
    "RejectedRecord",
    "ResumeRecord",
    "Split",
    "build_dataset",
    "load_corpus",
    "load_splits",
    "load_taxonomy",
]
