"""Training and model selection.

The protocol is fixed and deliberately conservative:

1. **Baselines first.** Majority-class and stratified-random. Every later number
   is reported as a delta against these; without them an accuracy figure is
   uninterpretable.
2. **Select on cross-validation over TRAIN.** Each (features x estimator) pair
   is scored by mean macro-F1 across stratified folds.
3. **Confirm on VALIDATION.** The best few candidates are refit on TRAIN and
   scored on the validation split.
4. **Tune the confidence policy on VALIDATION.** Never on test.
5. **Evaluate once on TEST.** One model, one evaluation, recorded.

The test split is opened exactly once, at the end. The production artifact is
fitted on TRAIN only -- deliberately not refitted on TRAIN+VALIDATION -- so that
the abstention threshold tuned on validation remains an out-of-sample estimate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from resume_classifier.config import Settings
from resume_classifier.dataset import Split, load_corpus, load_splits
from resume_classifier.dataset.manifest import git_commit, sha256_file, utc_now
from resume_classifier.dataset.schema import ResumeRecord
from resume_classifier.features.vectorizer import FEATURE_CONFIGS, FeatureConfig, build_vectorizer
from resume_classifier.inference.confidence import ConfidencePolicy, tune_policy
from resume_classifier.models.artifacts import next_version, save_bundle
from resume_classifier.models.evaluate import (
    confusion_matrix_dict,
    coverage_accuracy_curve,
    evaluate_predictions,
    text_classification_report,
    top_confusions,
)

logger = logging.getLogger(__name__)

CV_FOLDS = 5
#: How many distinct estimator families to hyperparameter-tune before choosing.
REFINE_TOP_FAMILIES = 2


@dataclass(frozen=True, slots=True)
class Candidate:
    """One estimator in the search space."""

    name: str
    build: Any  # callable returning a fresh estimator
    supports_probabilities: bool
    needs_calibration: bool = False


def _candidates(seed: int) -> tuple[Candidate, ...]:
    """The estimators compared during selection.

    No assumption is made about which wins; §19 of the project constitution
    requires the data to decide.
    """
    return (
        Candidate(
            "baseline_majority",
            lambda: DummyClassifier(strategy="most_frequent"),
            supports_probabilities=True,
        ),
        Candidate(
            "baseline_stratified",
            lambda: DummyClassifier(strategy="stratified", random_state=seed),
            supports_probabilities=True,
        ),
        Candidate("multinomial_nb", lambda: MultinomialNB(), supports_probabilities=True),
        Candidate("complement_nb", lambda: ComplementNB(), supports_probabilities=True),
        Candidate(
            "logistic_regression",
            lambda: LogisticRegression(
                max_iter=2000, C=5.0, class_weight="balanced", random_state=seed
            ),
            supports_probabilities=True,
        ),
        Candidate(
            "linear_svc",
            lambda: LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
            supports_probabilities=False,
            needs_calibration=True,
        ),
        Candidate(
            "sgd_modified_huber",
            lambda: SGDClassifier(
                loss="modified_huber",
                alpha=1e-5,
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ),
            supports_probabilities=True,
        ),
    )


def _split_records(
    records: Sequence[ResumeRecord], splits: dict[str, Split]
) -> tuple[list[ResumeRecord], list[ResumeRecord], list[ResumeRecord]]:
    train = [r for r in records if splits[r.record_id] is Split.TRAIN]
    validation = [r for r in records if splits[r.record_id] is Split.VALIDATION]
    test = [r for r in records if splits[r.record_id] is Split.TEST]
    return train, validation, test


def _xy(records: Sequence[ResumeRecord]) -> tuple[list[str], list[str]]:
    return [r.resume_text for r in records], [r.label for r in records]


def _build_pipeline(feature: FeatureConfig, candidate: Candidate, seed: int) -> Pipeline:
    estimator = candidate.build()
    if candidate.needs_calibration:
        estimator = CalibratedClassifierCV(
            estimator, method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=seed)
        )
    return Pipeline([("features", build_vectorizer(feature)), ("classifier", estimator)])


def _param_grid(feature: FeatureConfig, candidate: Candidate) -> dict[str, list[Any]]:
    """Hyperparameter grid for the refinement stage.

    Comparing families at fixed, arbitrary hyperparameters can crown the wrong
    winner, so the top families are re-compared after tuning rather than before.

    The parameter path depends on the pipeline shape: a calibrated estimator
    nests one level deeper, and a FeatureUnion nests its word vectorizer under
    ``word__``.
    """
    prefix = "classifier__estimator__" if candidate.needs_calibration else "classifier__"
    feature_prefix = "features__word__" if feature.use_char_ngrams else "features__"

    grid: dict[str, list[Any]] = {f"{feature_prefix}min_df": [1, 2, 3]}

    if candidate.name == "linear_svc":
        grid[f"{prefix}C"] = [0.1, 0.25, 0.5, 1.0, 2.0]
    elif candidate.name == "logistic_regression":
        grid[f"{prefix}C"] = [1.0, 5.0, 10.0, 25.0]
    elif candidate.name == "sgd_modified_huber":
        grid[f"{prefix}alpha"] = [1e-6, 1e-5, 1e-4]
    elif candidate.name in {"multinomial_nb", "complement_nb"}:
        grid[f"{prefix}alpha"] = [0.01, 0.1, 0.5, 1.0]
    return grid


def train(settings: Settings, *, strip_header_chars: int = 0) -> dict[str, Any]:
    """Run the full selection protocol and persist the winning model.

    Args:
        settings: Configuration supplying paths and the random seed.
        strip_header_chars: When > 0, remove this many leading characters from
            every document. Used for the header ablation, which measures how
            much the model leans on the job title printed at the top of a
            LiveCareer resume rather than on the body of the document.

    Returns:
        The metadata document that was written alongside the model.
    """
    records = load_corpus(settings.dataset_dir)
    splits = load_splits(settings.dataset_dir)
    train_records, validation_records, test_records = _split_records(records, splits)

    logger.info(
        "train=%d validation=%d test=%d",
        len(train_records),
        len(validation_records),
        len(test_records),
    )

    def prepare(texts: list[str]) -> list[str]:
        if strip_header_chars <= 0:
            return texts
        return [t[strip_header_chars:] for t in texts]

    x_train, y_train = _xy(train_records)
    x_validation, y_validation = _xy(validation_records)
    x_test, y_test = _xy(test_records)
    x_train, x_validation, x_test = (
        prepare(x_train),
        prepare(x_validation),
        prepare(x_test),
    )

    labels = sorted(set(y_train))
    seed = settings.random_seed

    # -- stage 1+2: cross-validated selection over TRAIN ------------------
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    search: list[dict[str, Any]] = []

    for feature in FEATURE_CONFIGS:
        for candidate in _candidates(seed):
            # Baselines ignore features; evaluate them once, not once per config.
            if candidate.name.startswith("baseline_") and feature is not FEATURE_CONFIGS[0]:
                continue
            pipeline = _build_pipeline(feature, candidate, seed)
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=folds, scoring="f1_macro", n_jobs=1
            )
            entry = {
                "features": feature.name,
                "estimator": candidate.name,
                "cv_macro_f1_mean": round(float(np.mean(scores)), 4),
                "cv_macro_f1_std": round(float(np.std(scores)), 4),
            }
            search.append(entry)
            logger.info(
                "cv %-20s %-22s macro-F1 %.4f (+/- %.4f)",
                feature.name,
                candidate.name,
                entry["cv_macro_f1_mean"],
                entry["cv_macro_f1_std"],
            )

    real = [s for s in search if not str(s["estimator"]).startswith("baseline_")]
    real.sort(key=lambda s: -float(s["cv_macro_f1_mean"]))

    feature_by_name = {f.name: f for f in FEATURE_CONFIGS}
    candidate_by_name = {c.name: c for c in _candidates(seed)}

    # -- stage 2b: tune the top families before crowning a winner ---------
    # The top entry at default hyperparameters is not necessarily the best
    # model; tuning can reorder the ranking. Refine the best configuration of
    # each of the top two distinct estimator families and re-compare.
    refinements: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    to_refine: list[dict[str, Any]] = []

    # Refine on word-only feature configurations. Character n-grams cost roughly
    # ten times as much to fit and, on this corpus, scored within noise of the
    # word-only equivalent -- so grid-searching a second family over them buys
    # nothing and makes training time unpredictable. The full comparison above
    # still includes them; only the tuning stage is restricted.
    word_only = {f.name for f in FEATURE_CONFIGS if not f.use_char_ngrams}
    affordable = [e for e in real if str(e["features"]) in word_only]

    for entry in affordable:
        family = str(entry["estimator"])
        if family in seen_families:
            continue
        seen_families.add(family)
        to_refine.append(entry)
        if len(to_refine) == REFINE_TOP_FAMILIES:
            break

    for entry in to_refine:
        feature = feature_by_name[str(entry["features"])]
        candidate = candidate_by_name[str(entry["estimator"])]
        grid = _param_grid(feature, candidate)
        search_cv = GridSearchCV(
            _build_pipeline(feature, candidate, seed),
            param_grid=grid,
            scoring="f1_macro",
            cv=folds,
            n_jobs=1,
            refit=False,
        )
        search_cv.fit(x_train, y_train)
        refinement = {
            "features": feature.name,
            "estimator": candidate.name,
            "best_params": dict(search_cv.best_params_),
            "cv_macro_f1_mean": round(float(search_cv.best_score_), 4),
            "baseline_cv_macro_f1": entry["cv_macro_f1_mean"],
        }
        refinements.append(refinement)
        logger.info(
            "refined %-20s %-22s macro-F1 %.4f (was %.4f) params=%s",
            feature.name,
            candidate.name,
            refinement["cv_macro_f1_mean"],
            entry["cv_macro_f1_mean"],
            refinement["best_params"],
        )

    refinements.sort(key=lambda r: -float(r["cv_macro_f1_mean"]))
    best = refinements[0]
    logger.info(
        "best after tuning: %s + %s (%s)",
        best["features"],
        best["estimator"],
        best["best_params"],
    )

    best_feature = feature_by_name[str(best["features"])]
    best_candidate = candidate_by_name[str(best["estimator"])]

    # -- stage 3: fit on TRAIN, confirm on VALIDATION ---------------------
    pipeline = _build_pipeline(best_feature, best_candidate, seed)
    pipeline.set_params(**best["best_params"])
    pipeline.fit(x_train, y_train)

    validation_pred = list(pipeline.predict(x_validation))
    validation_metrics = evaluate_predictions(y_validation, validation_pred, labels)
    logger.info("validation macro-F1 %.4f", validation_metrics["macro_f1"])

    # -- stage 4: tune the confidence policy on VALIDATION ----------------
    supports_proba = hasattr(pipeline, "predict_proba")
    if supports_proba:
        validation_proba = pipeline.predict_proba(x_validation)
        policy, policy_report = tune_policy(
            y_true=y_validation,
            probabilities=validation_proba,
            classes=list(pipeline.classes_),
        )
    else:  # pragma: no cover - every shipped candidate provides probabilities
        policy = ConfidencePolicy(threshold=0.0, margin=0.0)
        policy_report = {"note": "estimator exposes no probabilities; abstention disabled"}

    logger.info("confidence policy: threshold=%.2f margin=%.2f", policy.threshold, policy.margin)

    # -- stage 5: the test split, opened once -----------------------------
    test_pred = list(pipeline.predict(x_test))
    test_metrics = evaluate_predictions(y_test, test_pred, labels)
    logger.info(
        "TEST macro-F1 %.4f  accuracy %.4f", test_metrics["macro_f1"], test_metrics["accuracy"]
    )

    test_confidence: list[float] = []
    if supports_proba:
        test_proba = pipeline.predict_proba(x_test)
        test_confidence = [float(row.max()) for row in test_proba]

    baselines = {
        str(s["estimator"]): s["cv_macro_f1_mean"]
        for s in search
        if str(s["estimator"]).startswith("baseline_")
    }

    dataset_manifest = settings.dataset_dir / "MANIFEST.json"
    metadata: dict[str, Any] = {
        "created_at": utc_now(),
        "git_commit": git_commit(settings.project_root),
        "labels": labels,
        "supports_probabilities": supports_proba,
        "selected": {
            "features": best_feature.describe(),
            "estimator": best_candidate.name,
            "hyperparameters": {str(k): v for k, v in best["best_params"].items()},
        },
        "confidence": {
            "threshold": policy.threshold,
            "margin": policy.margin,
            "tuned_on": "validation",
            "report": policy_report,
        },
        "dataset": {
            "version": settings.dataset_version,
            "manifest_sha256": (
                sha256_file(dataset_manifest) if dataset_manifest.is_file() else None
            ),
            "counts": {
                "train": len(train_records),
                "validation": len(validation_records),
                "test": len(test_records),
            },
            "strip_header_chars": strip_header_chars,
        },
        "training": {
            "seed": seed,
            "cv_folds": CV_FOLDS,
            "search": search,
            "refinements": refinements,
            "baselines_cv_macro_f1": baselines,
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
            "test_confusions": top_confusions(y_test, test_pred, labels, limit=12),
            "test_confusion_matrix": confusion_matrix_dict(y_test, test_pred, labels),
            "test_classification_report": text_classification_report(y_test, test_pred, labels),
            "test_coverage_curve": (
                coverage_accuracy_curve(
                    y_test,
                    test_pred,
                    test_confidence,
                    thresholds=[0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                )
                if test_confidence
                else []
            ),
        },
    }

    # An ablation is a measurement, not a release. Writing it into the version
    # namespace would make it the "newest" model and silently take over serving,
    # so it goes into a sibling directory that `list_versions` does not scan.
    if strip_header_chars > 0:
        version = f"strip{strip_header_chars}"
        target_dir = settings.models_dir / "ablations"
        logger.info("ablation run: saving outside the serving version namespace")
    else:
        version = next_version(settings.models_dir)
        target_dir = settings.models_dir

    save_bundle(target_dir, version, pipeline, metadata)
    metadata["version"] = version
    return metadata
