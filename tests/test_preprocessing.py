"""Tests for the shared text normalisation.

This module is imported by both the training pipeline and the serving pipeline,
so its behaviour is the contract that keeps them in agreement.
"""

from __future__ import annotations

import pytest

from resume_classifier.preprocessing.text import (
    feature_normalize,
    ingestion_normalize,
    redact_pii,
    word_count,
)


class TestIngestionNormalize:
    def test_empty_input_is_safe(self) -> None:
        assert ingestion_normalize("") == ""

    def test_normalises_line_endings(self) -> None:
        assert "\r" not in ingestion_normalize("a\r\nb\rc")

    def test_collapses_horizontal_whitespace_but_keeps_lines(self) -> None:
        assert ingestion_normalize("a    b\nc") == "a b\nc"

    def test_limits_runs_of_blank_lines(self) -> None:
        assert ingestion_normalize("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_case_and_punctuation(self) -> None:
        text = "Senior Engineer, C++ (2019-2024)."
        assert ingestion_normalize(text) == text

    def test_strips_control_and_zero_width_characters(self) -> None:
        result = ingestion_normalize("a​b\x00c")
        assert "​" not in result
        assert "\x00" not in result

    def test_is_idempotent(self) -> None:
        text = "Some\r\n\r\n\r\nmessy    text​ here."
        once = ingestion_normalize(text)
        assert ingestion_normalize(once) == once


class TestRedactPii:
    def test_redacts_email(self) -> None:
        assert "john.smith@example.com" not in redact_pii("mail john.smith@example.com now")

    def test_redacts_url(self) -> None:
        assert "linkedin" not in redact_pii("see https://linkedin.com/in/js")

    def test_redacts_phone(self) -> None:
        assert "5551234567" not in redact_pii("call 555 123 4567")


class TestFeatureNormalize:
    def test_empty_input_is_safe(self) -> None:
        assert feature_normalize("") == ""

    def test_is_idempotent(self) -> None:
        text = "Python, Node.js, C++ -- 15 years. a.b@c.io"
        once = feature_normalize(text)
        assert feature_normalize(once) == once

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("C++", "c++"),
            ("C#", "c#"),
            ("F#", "f#"),
            ("Node.js", "node.js"),
            ("back-end", "back-end"),
        ],
    )
    def test_preserves_meaningful_punctuation(self, token: str, expected: str) -> None:
        """A naive stripper turns "c++" into "c" and loses the skill entirely."""
        assert expected in feature_normalize(f"skilled in {token} development")

    def test_strips_noise_punctuation(self) -> None:
        result = feature_normalize("a --- b ... c.")
        assert result == "a b c"

    def test_lowercases(self) -> None:
        assert feature_normalize("PYTHON") == "python"

    def test_replaces_numbers_with_a_token(self) -> None:
        assert "2019" not in feature_normalize("from 2019 to 2024")

    def test_redacts_identifiers(self) -> None:
        result = feature_normalize("John at john@x.io or 555-123-4567")
        assert "john@x.io" not in result
        assert "emailaddr" in result

    def test_newline_delimited_text_yields_tokens(self) -> None:
        """Resume skill lists are newline-delimited; they must survive."""
        assert feature_normalize("Python\nDocker\nSQL") == "python docker sql"


def test_word_count() -> None:
    assert word_count("one two three") == 3
    assert word_count("") == 0
