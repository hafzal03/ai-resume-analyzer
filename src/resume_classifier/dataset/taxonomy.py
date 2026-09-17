"""Category taxonomy: loading, validation, and raw-label mapping.

The taxonomy is a version-controlled TOML file, not code. It is read with the
standard library's ``tomllib``, so curating the class list costs no dependency.

Every raw label a source corpus provides must map to exactly one canonical
class. An unrecognised raw label is a build failure, never a silent drop -- a
corpus that grows a new category should force a taxonomy decision rather than
quietly losing rows.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class TaxonomyError(ValueError):
    """The taxonomy file is malformed, inconsistent, or self-contradictory."""


@dataclass(frozen=True, slots=True)
class CategoryClass:
    """One canonical class in the taxonomy."""

    id: str
    display: str
    description: str
    source_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CategoryTaxonomy:
    """A closed set of canonical classes plus the raw-label mapping."""

    version: int
    description: str
    classes: tuple[CategoryClass, ...]
    _by_source_label: Mapping[str, str]

    @property
    def class_ids(self) -> tuple[str, ...]:
        """Canonical class ids, sorted."""
        return tuple(sorted(c.id for c in self.classes))

    @property
    def display_names(self) -> Mapping[str, str]:
        """Canonical id -> human-readable display name."""
        return {c.id: c.display for c in self.classes}

    def map_label(self, raw_label: str) -> str | None:
        """Map a source label to its canonical class id, or ``None`` if unknown."""
        return self._by_source_label.get(_normalise_source_label(raw_label))

    def describe(self, class_id: str) -> str:
        """Display name for a canonical class id, falling back to the id."""
        return self.display_names.get(class_id, class_id)


def _normalise_source_label(raw: str) -> str:
    """Compare source labels case-insensitively and whitespace-insensitively."""
    return " ".join(raw.strip().split()).upper()


def load_taxonomy(path: Path) -> CategoryTaxonomy:
    """Load and validate a taxonomy TOML file.

    Raises:
        TaxonomyError: on a missing version, duplicate class id, duplicate
            source label, or empty class list.
    """
    if not path.is_file():
        msg = f"taxonomy file not found: {path}"
        raise TaxonomyError(msg)

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    version = data.get("taxonomy_version")
    if not isinstance(version, int):
        msg = "taxonomy_version must be an integer"
        raise TaxonomyError(msg)

    raw_classes = data.get("classes") or []
    if not raw_classes:
        msg = "taxonomy defines no classes"
        raise TaxonomyError(msg)

    classes: list[CategoryClass] = []
    by_source_label: dict[str, str] = {}
    seen_ids: set[str] = set()

    for entry in raw_classes:
        class_id = str(entry["id"]).strip()
        if not class_id:
            msg = "class with an empty id"
            raise TaxonomyError(msg)
        if class_id in seen_ids:
            msg = f"duplicate class id: {class_id}"
            raise TaxonomyError(msg)
        seen_ids.add(class_id)

        source_labels = tuple(str(s) for s in (entry.get("source_labels") or ()))
        if not source_labels:
            msg = f"class {class_id!r} maps no source labels"
            raise TaxonomyError(msg)

        for label in source_labels:
            key = _normalise_source_label(label)
            if key in by_source_label:
                msg = (
                    f"source label {label!r} is claimed by both "
                    f"{by_source_label[key]!r} and {class_id!r}"
                )
                raise TaxonomyError(msg)
            by_source_label[key] = class_id

        classes.append(
            CategoryClass(
                id=class_id,
                display=str(entry.get("display", class_id)),
                description=str(entry.get("description", "")),
                source_labels=source_labels,
            )
        )

    return CategoryTaxonomy(
        version=version,
        description=str(data.get("description", "")),
        classes=tuple(sorted(classes, key=lambda c: c.id)),
        _by_source_label=by_source_label,
    )
