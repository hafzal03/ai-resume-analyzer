"""Skill taxonomy: loading and validation.

Like the category taxonomy, this is a version-controlled TOML file read with the
standard library. A skill has one canonical id, one display name, a category,
and the set of surface forms (aliases) that indicate its presence.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class SkillTaxonomyError(ValueError):
    """The skill taxonomy file is malformed or self-contradictory."""


@dataclass(frozen=True, slots=True)
class Skill:
    """One canonical skill and the surface forms that indicate it."""

    id: str
    display: str
    category: str
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SkillTaxonomy:
    """A controlled vocabulary of skills."""

    version: int
    description: str
    skills: tuple[Skill, ...]

    @property
    def by_id(self) -> Mapping[str, Skill]:
        """Canonical id -> skill."""
        return {s.id: s for s in self.skills}

    @property
    def categories(self) -> tuple[str, ...]:
        """Distinct skill categories, sorted."""
        return tuple(sorted({s.category for s in self.skills}))

    def alias_pairs(self) -> list[tuple[str, str]]:
        """Every ``(alias, skill_id)`` pair, longest alias first.

        Longest-first ordering matters for matching: "machine learning" must win
        over a hypothetical "learning", and "node.js" over "node".
        """
        pairs = [(alias, skill.id) for skill in self.skills for alias in skill.aliases]
        pairs.sort(key=lambda pair: (-len(pair[0]), pair[0]))
        return pairs


def load_skill_taxonomy(path: Path) -> SkillTaxonomy:
    """Load and validate a skill taxonomy TOML file.

    Raises:
        SkillTaxonomyError: on a missing version, duplicate skill id, duplicate
            alias, or an empty alias list.
    """
    if not path.is_file():
        msg = f"skill taxonomy not found: {path}"
        raise SkillTaxonomyError(msg)

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    version = data.get("taxonomy_version")
    if not isinstance(version, int):
        msg = "taxonomy_version must be an integer"
        raise SkillTaxonomyError(msg)

    entries = data.get("skills") or []
    if not entries:
        msg = "skill taxonomy defines no skills"
        raise SkillTaxonomyError(msg)

    skills: list[Skill] = []
    seen_ids: set[str] = set()
    alias_owner: dict[str, str] = {}

    for entry in entries:
        skill_id = str(entry["id"]).strip()
        if not skill_id:
            msg = "skill with an empty id"
            raise SkillTaxonomyError(msg)
        if skill_id in seen_ids:
            msg = f"duplicate skill id: {skill_id}"
            raise SkillTaxonomyError(msg)
        seen_ids.add(skill_id)

        aliases = tuple(" ".join(str(a).casefold().split()) for a in (entry.get("aliases") or ()))
        aliases = tuple(a for a in aliases if a)
        if not aliases:
            msg = f"skill {skill_id!r} declares no aliases"
            raise SkillTaxonomyError(msg)

        for alias in aliases:
            if alias in alias_owner:
                msg = f"alias {alias!r} is claimed by both {alias_owner[alias]!r} and {skill_id!r}"
                raise SkillTaxonomyError(msg)
            alias_owner[alias] = skill_id

        skills.append(
            Skill(
                id=skill_id,
                display=str(entry.get("display", skill_id)),
                category=str(entry.get("category", "general")),
                aliases=aliases,
            )
        )

    return SkillTaxonomy(
        version=version,
        description=str(data.get("description", "")),
        skills=tuple(sorted(skills, key=lambda s: s.id)),
    )
