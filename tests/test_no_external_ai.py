"""The project must never depend on an external AI or model provider.

This is a hard architectural constraint, so it is a test rather than a
convention. It checks three things: no provider SDK is importable, no provider
appears in the declared dependencies, and no source file references one.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from resume_classifier.config import PROJECT_ROOT

#: Packages that would place an external model provider in the request path.
FORBIDDEN_PACKAGES = (
    "openai",
    "anthropic",
    "google.generativeai",
    "google_generativeai",
    "mistralai",
    "groq",
    "cohere",
    "langchain",
    "langchain_core",
    "langgraph",
    "llama_index",
    "llamaindex",
    "transformers",
    "torch",
    "tensorflow",
    "sentence_transformers",
)

FORBIDDEN_PATTERN = re.compile(
    r"\b(openai|anthropic|claude|gemini|mistralai|groq|cohere|langchain|langgraph"
    r"|llama_?index|sentence_transformers|huggingface_hub)\b",
    re.IGNORECASE,
)

SOURCE_ROOT = PROJECT_ROOT / "src" / "resume_classifier"


@pytest.mark.parametrize("package", FORBIDDEN_PACKAGES)
def test_provider_sdk_is_not_installed(package: str) -> None:
    """None of these may be importable in the runtime environment."""
    try:
        found = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        found = False
    assert not found, f"{package} must not be installed; the classifier runs locally"


def test_declared_dependencies_contain_no_provider() -> None:
    """A provider must not creep into pyproject.toml either."""
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared: list[str] = list(data["project"].get("dependencies", []))
    for extra in (data["project"].get("optional-dependencies") or {}).values():
        declared.extend(extra)

    offenders = [d for d in declared if FORBIDDEN_PATTERN.search(d)]
    assert not offenders, f"forbidden dependency declared: {offenders}"


def test_source_contains_no_provider_reference() -> None:
    """No module may name a provider, even in a comment."""
    offenders: list[str] = []
    for path in SOURCE_ROOT.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN_PATTERN.search(line):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}: {line.strip()}")
    assert not offenders, "provider references found:\n" + "\n".join(offenders)


def test_runtime_makes_no_network_calls() -> None:
    """The serving path must not reach the network at all."""
    network_pattern = re.compile(r"\b(requests\.|httpx\.|urlopen|aiohttp|socket\.socket)\b")
    runtime_dirs = ("api", "inference", "extraction", "preprocessing", "skills", "db")

    offenders: list[str] = []
    for directory in runtime_dirs:
        for path in (SOURCE_ROOT / directory).rglob("*.py"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if network_pattern.search(line):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{number}")
    assert not offenders, f"network access in the serving path: {offenders}"


def test_classification_works_offline(taxonomy_dir: Path) -> None:
    """Sanity check: skill extraction is pure computation, no I/O beyond the file."""
    from resume_classifier.skills.extractor import SkillExtractor
    from resume_classifier.skills.taxonomy import load_skill_taxonomy

    extractor = SkillExtractor(load_skill_taxonomy(taxonomy_dir / "skills_v1.toml"))
    assert "python" in extractor.extract_ids("Experienced Python developer")
