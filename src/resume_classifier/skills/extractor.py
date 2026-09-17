"""Deterministic skill extraction.

No model and no inference: a skill is present if and only if one of its aliases
appears in the normalised resume text as a whole token sequence. The same input
always produces the same output, and any result can be explained by pointing at
the alias that matched.

This replaces the naive ``" skill " in " text "`` idiom, which silently returns
nothing for newline- or comma-delimited skill lists -- that is, for most real
resumes.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from resume_classifier.preprocessing.text import feature_normalize
from resume_classifier.skills.taxonomy import SkillTaxonomy, load_skill_taxonomy

# A token boundary for text that has been through `feature_normalize`. Letters,
# digits and the symbols that survive normalisation all count as "inside a
# token", so "c++" does not match inside "c++builder" and "sql" does not match
# inside "mysql".
_BOUNDARY_CHARS = "a-z0-9+#"


@dataclass(frozen=True, slots=True)
class ExtractedSkill:
    """One skill found in a document."""

    id: str
    display: str
    category: str
    matched_alias: str
    occurrences: int


class SkillExtractor:
    """Matches a controlled skill vocabulary against resume text.

    All aliases are compiled into a single alternation, so extraction is one
    regex pass over the document rather than one pass per skill.
    """

    def __init__(self, taxonomy: SkillTaxonomy) -> None:
        self._taxonomy = taxonomy
        self._alias_to_skill = dict(taxonomy.alias_pairs())
        # Longest alias first so the alternation prefers the most specific match.
        alternation = "|".join(re.escape(alias) for alias, _ in taxonomy.alias_pairs())
        self._pattern = re.compile(
            rf"(?<![{_BOUNDARY_CHARS}])(?:{alternation})(?![{_BOUNDARY_CHARS}])"
        )

    @property
    def taxonomy(self) -> SkillTaxonomy:
        """The vocabulary this extractor matches against."""
        return self._taxonomy

    @property
    def skill_ids(self) -> tuple[str, ...]:
        """All canonical skill ids, sorted -- the feature space for skills."""
        return tuple(sorted(s.id for s in self._taxonomy.skills))

    def extract(self, text: str, *, already_normalized: bool = False) -> list[ExtractedSkill]:
        """Return the skills present in ``text``, sorted by frequency then name.

        Args:
            text: Resume text.
            already_normalized: Set when ``text`` has already been through
                :func:`feature_normalize`, to avoid normalising twice.
        """
        if not text:
            return []
        normalized = text if already_normalized else feature_normalize(text)

        counts: dict[str, int] = {}
        first_alias: dict[str, str] = {}
        for match in self._pattern.finditer(normalized):
            alias = match.group(0)
            skill_id = self._alias_to_skill[alias]
            counts[skill_id] = counts.get(skill_id, 0) + 1
            first_alias.setdefault(skill_id, alias)

        by_id = self._taxonomy.by_id
        found = [
            ExtractedSkill(
                id=skill_id,
                display=by_id[skill_id].display,
                category=by_id[skill_id].category,
                matched_alias=first_alias[skill_id],
                occurrences=count,
            )
            for skill_id, count in counts.items()
        ]
        found.sort(key=lambda s: (-s.occurrences, s.display))
        return found

    def extract_ids(self, text: str, *, already_normalized: bool = False) -> list[str]:
        """Return just the canonical skill ids present, sorted."""
        return sorted(s.id for s in self.extract(text, already_normalized=already_normalized))

    def binary_vector(self, texts: Sequence[str]) -> list[list[int]]:
        """Presence/absence matrix over :attr:`skill_ids`, for use as features."""
        index = {skill_id: i for i, skill_id in enumerate(self.skill_ids)}
        rows: list[list[int]] = []
        for text in texts:
            row = [0] * len(index)
            for skill_id in self.extract_ids(text):
                row[index[skill_id]] = 1
            rows.append(row)
        return rows


@lru_cache(maxsize=4)
def load_extractor(path: Path) -> SkillExtractor:
    """Load a skill extractor from a taxonomy file, cached by path."""
    return SkillExtractor(load_skill_taxonomy(path))
