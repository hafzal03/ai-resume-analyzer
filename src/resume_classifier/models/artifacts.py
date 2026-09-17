"""Versioned model artifacts.

A model is never a bare pickle. Each version is a directory containing the
fitted pipeline plus a metadata document recording everything needed to trust
and reproduce it: dataset hashes, taxonomy versions, seed, library versions, the
full metric set, and the confidence policy that was tuned alongside it.

Loading verifies that metadata against the running environment. The legacy
system shipped a vectorizer trained on a different dataset than the one in the
repository and nothing noticed; here that is a loud failure.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

PIPELINE_FILENAME = "pipeline.joblib"
METADATA_FILENAME = "metadata.json"

_VERSION_PATTERN = re.compile(r"^v(\d+)$")


class ModelNotFoundError(FileNotFoundError):
    """No usable model artifact exists."""


class ModelCompatibilityError(RuntimeError):
    """A stored artifact does not match the running environment."""


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """A fitted pipeline together with its provenance and metrics."""

    version: str
    pipeline: Pipeline
    metadata: dict[str, Any]

    @property
    def labels(self) -> list[str]:
        """Canonical class ids this model can emit, in a stable order."""
        labels: list[str] = list(self.metadata["labels"])
        return labels

    @property
    def confidence_threshold(self) -> float:
        """Minimum top-class probability required to commit to a prediction."""
        return float(self.metadata["confidence"]["threshold"])

    @property
    def margin_threshold(self) -> float:
        """Minimum gap between the top two classes required to commit."""
        return float(self.metadata["confidence"]["margin"])

    @property
    def supports_probabilities(self) -> bool:
        """Whether the estimator exposes calibrated ``predict_proba``."""
        return bool(self.metadata.get("supports_probabilities", False))

    def test_macro_f1(self) -> float | None:
        """Held-out macro-F1, if the metadata records a test evaluation."""
        test = self.metadata.get("metrics", {}).get("test")
        return None if test is None else float(test["macro_f1"])


def next_version(models_dir: Path) -> str:
    """Return the next unused ``vN`` directory name."""
    highest = 0
    if models_dir.is_dir():
        for child in models_dir.iterdir():
            match = _VERSION_PATTERN.match(child.name)
            if match and child.is_dir():
                highest = max(highest, int(match.group(1)))
    return f"v{highest + 1}"


def list_versions(models_dir: Path) -> list[str]:
    """All model versions present, oldest first."""
    if not models_dir.is_dir():
        return []
    versions = [
        child.name
        for child in models_dir.iterdir()
        if child.is_dir()
        and _VERSION_PATTERN.match(child.name)
        and (child / PIPELINE_FILENAME).is_file()
    ]
    return sorted(versions, key=lambda v: int(v[1:]))


def save_bundle(
    models_dir: Path,
    version: str,
    pipeline: Pipeline,
    metadata: dict[str, Any],
) -> Path:
    """Persist a fitted pipeline and its metadata under ``models_dir/version``."""
    target = models_dir / version
    target.mkdir(parents=True, exist_ok=True)

    enriched = {
        **metadata,
        "version": version,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(pipeline, target / PIPELINE_FILENAME, compress=3)
    (target / METADATA_FILENAME).write_text(
        json.dumps(enriched, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("saved model %s to %s", version, target)
    return target


def load_bundle(models_dir: Path, version: str | None = None) -> ModelBundle:
    """Load a model artifact, defaulting to the newest version.

    Raises:
        ModelNotFoundError: if no artifact exists.
        ModelCompatibilityError: if the artifact is structurally unusable.
    """
    available = list_versions(models_dir)
    if not available:
        msg = f"no trained model found in {models_dir}. Run 'rc-train' to train one."
        raise ModelNotFoundError(msg)

    resolved = version or available[-1]
    if resolved not in available:
        msg = f"model version {resolved!r} not found; available: {', '.join(available)}"
        raise ModelNotFoundError(msg)

    target = models_dir / resolved
    metadata = json.loads((target / METADATA_FILENAME).read_text(encoding="utf-8"))
    pipeline = joblib.load(target / PIPELINE_FILENAME)

    stored_sklearn = metadata.get("sklearn_version")
    if stored_sklearn and stored_sklearn != sklearn.__version__:
        # A mismatch is not automatically fatal, but it must never be silent:
        # this is exactly how the previous system shipped a stale model.
        logger.warning(
            "model %s was trained with scikit-learn %s but %s is installed; "
            "retrain if predictions look wrong",
            resolved,
            stored_sklearn,
            sklearn.__version__,
        )

    if "labels" not in metadata or "confidence" not in metadata:
        msg = f"model {resolved} has incomplete metadata; retrain it"
        raise ModelCompatibilityError(msg)

    return ModelBundle(version=resolved, pipeline=pipeline, metadata=metadata)
