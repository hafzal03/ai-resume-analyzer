"""The runtime classification pipeline.

This is the only object the web layer talks to. It owns the loaded model, the
skill extractor, and the confidence policy, and it turns resume text into a
result -- with no knowledge of HTTP, templates, or the database.

Training code is deliberately not importable from here: the serving process has
no reason to carry pandas or the experiment harness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from resume_classifier.config import Settings
from resume_classifier.dataset.taxonomy import CategoryTaxonomy, load_taxonomy
from resume_classifier.inference.confidence import ConfidencePolicy, Verdict, apply_policy
from resume_classifier.models.artifacts import ModelBundle, load_bundle
from resume_classifier.preprocessing.text import feature_normalize, word_count
from resume_classifier.skills.extractor import ExtractedSkill, SkillExtractor, load_extractor

logger = logging.getLogger(__name__)


class TextTooShortError(ValueError):
    """The supplied text is too short to classify meaningfully."""

    code = "TEXT_TOO_SHORT"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Everything the system concluded about one resume."""

    label: str | None
    display_label: str
    confidence: float
    margin: float
    is_uncertain: bool
    alternatives: list[dict[str, Any]]
    skills: list[ExtractedSkill]
    model_version: str
    text_char_count: int
    text_word_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, used by the API and the templates."""
        return {
            "prediction": {
                "label": self.label,
                "display": self.display_label,
                "confidence": self.confidence,
                "margin": self.margin,
                "is_uncertain": self.is_uncertain,
            },
            "alternatives": self.alternatives,
            "skills": [
                {
                    "id": s.id,
                    "display": s.display,
                    "category": s.category,
                    "occurrences": s.occurrences,
                }
                for s in self.skills
            ],
            "skill_count": len(self.skills),
            "document": {
                "characters": self.text_char_count,
                "words": self.text_word_count,
            },
            "model_version": self.model_version,
            "warnings": self.warnings,
        }


@dataclass
class ClassificationEngine:
    """Loads a model once and classifies many documents."""

    settings: Settings
    bundle: ModelBundle
    extractor: SkillExtractor
    taxonomy: CategoryTaxonomy

    @classmethod
    def load(cls, settings: Settings) -> ClassificationEngine:
        """Construct an engine from configured paths.

        Raises:
            ModelNotFoundError: if no trained model is available.
        """
        bundle = load_bundle(settings.models_dir, settings.model_version)
        extractor = load_extractor(settings.taxonomy_dir / "skills_v1.toml")
        taxonomy = load_taxonomy(settings.taxonomy_dir / "categories_v1.toml")
        logger.info(
            "loaded model %s (test macro-F1 %s) with %d skills",
            bundle.version,
            bundle.test_macro_f1(),
            len(extractor.skill_ids),
        )
        return cls(settings=settings, bundle=bundle, extractor=extractor, taxonomy=taxonomy)

    @cached_property
    def policy(self) -> ConfidencePolicy:
        """The abstention policy tuned during training."""
        return ConfidencePolicy(
            threshold=self.bundle.confidence_threshold,
            margin=self.bundle.margin_threshold,
        )

    @property
    def model_version(self) -> str:
        """Version string of the loaded model."""
        return self.bundle.version

    def classify(self, text: str, *, min_words: int = 30) -> ClassificationResult:
        """Classify one resume and extract its skills.

        Args:
            text: Resume text, already extracted from whatever carried it.
            min_words: Refuse to classify anything shorter than this.

        Raises:
            TextTooShortError: when the document has too little content for the
                prediction to mean anything.
        """
        normalized = feature_normalize(text)
        words = word_count(normalized)
        if words < min_words:
            msg = (
                f"document has {words} usable words, at least {min_words} are "
                "required for a meaningful classification"
            )
            raise TextTooShortError(msg)

        warnings: list[str] = []
        classes = [str(c) for c in self.bundle.pipeline.classes_]

        if self.bundle.supports_probabilities:
            probabilities = self.bundle.pipeline.predict_proba([text])[0]
            verdict = apply_policy(probabilities, classes, self.policy)
        else:  # pragma: no cover - shipped models all support probabilities
            predicted = str(self.bundle.pipeline.predict([text])[0])
            verdict = Verdict(
                label=predicted,
                confidence=0.0,
                margin=0.0,
                is_uncertain=False,
                ranked=[(predicted, 0.0)],
            )
            warnings.append("model does not expose probabilities; confidence unavailable")

        if verdict.is_uncertain:
            warnings.append(
                "confidence below the configured threshold; treat this as a "
                "suggestion rather than a classification"
            )

        skills = self.extractor.extract(normalized, already_normalized=True)
        if not skills:
            warnings.append("no known skills were recognised in this document")

        return ClassificationResult(
            label=verdict.label,
            display_label=(
                "Uncertain"
                if verdict.is_uncertain or verdict.label is None
                else self.taxonomy.describe(verdict.label)
            ),
            confidence=verdict.confidence,
            margin=verdict.margin,
            is_uncertain=verdict.is_uncertain,
            alternatives=[
                {
                    "label": label,
                    "display": self.taxonomy.describe(label),
                    "probability": probability,
                }
                for label, probability in verdict.ranked
            ],
            skills=skills,
            model_version=self.bundle.version,
            text_char_count=len(text),
            text_word_count=words,
            warnings=warnings,
        )
