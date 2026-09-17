"""Tests for deterministic skill extraction.

The cases below are drawn from the ways the previous implementation failed: it
used ``f" {skill} " in f" {text} "``, which returns nothing whenever skills are
separated by newlines, commas or pipes -- that is, on most real resumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resume_classifier.skills.extractor import SkillExtractor
from resume_classifier.skills.taxonomy import (
    Skill,
    SkillTaxonomy,
    SkillTaxonomyError,
    load_skill_taxonomy,
)


@pytest.fixture
def extractor(taxonomy_dir: Path) -> SkillExtractor:
    return SkillExtractor(load_skill_taxonomy(taxonomy_dir / "skills_v1.toml"))


class TestTaxonomyValidation:
    def test_real_taxonomy_loads(self, taxonomy_dir: Path) -> None:
        taxonomy = load_skill_taxonomy(taxonomy_dir / "skills_v1.toml")
        assert taxonomy.version == 1
        assert len(taxonomy.skills) > 100

    def test_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(SkillTaxonomyError, match="not found"):
            load_skill_taxonomy(tmp_path / "nope.toml")

    def test_duplicate_alias_is_rejected(self, tmp_path: Path) -> None:
        """Two skills claiming one alias makes extraction ambiguous."""
        path = tmp_path / "dupe.toml"
        path.write_text(
            "taxonomy_version = 1\n"
            '[[skills]]\nid = "a"\naliases = ["shared"]\n'
            '[[skills]]\nid = "b"\naliases = ["shared"]\n',
            encoding="utf-8",
        )
        with pytest.raises(SkillTaxonomyError, match="claimed by both"):
            load_skill_taxonomy(path)

    def test_skill_without_aliases_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.toml"
        path.write_text(
            'taxonomy_version = 1\n[[skills]]\nid = "a"\naliases = []\n', encoding="utf-8"
        )
        with pytest.raises(SkillTaxonomyError, match="no aliases"):
            load_skill_taxonomy(path)

    def test_longest_aliases_are_tried_first(self, taxonomy_dir: Path) -> None:
        taxonomy = load_skill_taxonomy(taxonomy_dir / "skills_v1.toml")
        lengths = [len(alias) for alias, _ in taxonomy.alias_pairs()]
        assert lengths == sorted(lengths, reverse=True)


class TestExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Skills:\nPython\nDocker\nSQL", {"python", "docker", "sql"}),
            ("Python, Docker and AWS.", {"python", "docker", "aws"}),
            ("Python | Docker | AWS", {"python", "docker", "aws"}),
            ("PYTHON and DOCKER", {"python", "docker"}),
            ("Proficient in C++, C#", {"cpp", "csharp"}),
        ],
    )
    def test_delimiters_do_not_defeat_matching(
        self, extractor: SkillExtractor, text: str, expected: set[str]
    ) -> None:
        assert expected.issubset(set(extractor.extract_ids(text)))

    def test_empty_text_yields_nothing(self, extractor: SkillExtractor) -> None:
        assert extractor.extract("") == []

    def test_no_substring_false_positives(self, extractor: SkillExtractor) -> None:
        """ "sql" must not fire inside "mysql"; token boundaries are enforced."""
        found = set(extractor.extract_ids("We run mysql here"))
        assert "mysql" in found
        assert "sql" not in found

    def test_occurrences_are_counted(self, extractor: SkillExtractor) -> None:
        skills = extractor.extract("python python python")
        assert skills[0].id == "python"
        assert skills[0].occurrences == 3

    def test_results_are_sorted_by_frequency(self, extractor: SkillExtractor) -> None:
        skills = extractor.extract("docker docker docker python")
        assert next(s.id for s in skills) == "docker"

    def test_is_deterministic(self, extractor: SkillExtractor) -> None:
        text = "Python, Docker, SQL, project management, customer service"
        assert extractor.extract_ids(text) == extractor.extract_ids(text)

    def test_covers_non_technical_domains(self, extractor: SkillExtractor) -> None:
        """The corpus spans 24 industries; an IT-only vocabulary would be useless."""
        culinary = extractor.extract_ids("ServSafe certified, menu planning, sous chef")
        clinical = extractor.extract_ids("HIPAA, patient care, phlebotomy")
        assert culinary
        assert clinical

    def test_binary_vector_shape(self, extractor: SkillExtractor) -> None:
        rows = extractor.binary_vector(["python", "docker"])
        assert len(rows) == 2
        assert all(len(row) == len(extractor.skill_ids) for row in rows)


def test_extractor_handles_single_skill_taxonomy() -> None:
    """A minimal taxonomy must compile to a valid pattern."""
    taxonomy = SkillTaxonomy(
        version=1,
        description="",
        skills=(Skill(id="only", display="Only", category="x", aliases=("only",)),),
    )
    assert SkillExtractor(taxonomy).extract_ids("the only one") == ["only"]
