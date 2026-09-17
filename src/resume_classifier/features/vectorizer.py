"""Feature engineering.

Every vectorizer is constructed with ``preprocessor=feature_normalize``. That is
the mechanism which guarantees training and inference see identical text: the
normalisation travels *inside* the fitted artifact, so it cannot drift away from
whatever the serving code happens to do.

Because the preprocessor has already produced clean, space-delimited tokens, the
token pattern is simply "a run of non-whitespace". The scikit-learn default
(``\\b\\w\\w+\\b``) would silently discard ``c++``, ``c#`` and ``node.js``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from resume_classifier.preprocessing.text import feature_normalize

#: Tokens are whitespace-delimited because the preprocessor guarantees it.
TOKEN_PATTERN = r"(?u)\S+"


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """One point in the feature search space."""

    name: str
    word_ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.9
    sublinear_tf: bool = True
    use_char_ngrams: bool = False
    char_ngram_range: tuple[int, int] = (3, 5)
    max_features: int | None = None

    def describe(self) -> dict[str, object]:
        """Serialisable description, recorded in model metadata."""
        return {
            "name": self.name,
            "word_ngram_max": self.word_ngram_max,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "sublinear_tf": self.sublinear_tf,
            "use_char_ngrams": self.use_char_ngrams,
            "char_ngram_range": list(self.char_ngram_range),
            "max_features": self.max_features,
        }


def build_vectorizer(config: FeatureConfig) -> TfidfVectorizer | FeatureUnion:
    """Construct the (unfitted) vectorizer described by ``config``."""
    word = TfidfVectorizer(
        preprocessor=feature_normalize,
        lowercase=False,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, config.word_ngram_max),
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        max_features=config.max_features,
        strip_accents=None,
    )
    if not config.use_char_ngrams:
        return word

    char = TfidfVectorizer(
        preprocessor=feature_normalize,
        lowercase=False,
        analyzer="char_wb",
        ngram_range=config.char_ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        sublinear_tf=config.sublinear_tf,
        max_features=200_000,
    )
    return FeatureUnion([("word", word), ("char", char)])


#: The feature configurations compared during model selection. Nothing here is
#: assumed to be best; the numbers decide.
FEATURE_CONFIGS: tuple[FeatureConfig, ...] = (
    FeatureConfig(name="word_1gram", word_ngram_max=1),
    FeatureConfig(name="word_1_2gram", word_ngram_max=2),
    FeatureConfig(name="word_1_2gram_char", word_ngram_max=2, use_char_ngrams=True),
)
