"""Terms file loading tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from stt.terms import TermsError, load_terms_file, terms_flag_was_explicit


def test_load_missing_optional(tmp_path: Path):
    context, terms = load_terms_file(tmp_path / "terms.txt", required=False)
    assert context == ""
    assert terms == []


def test_load_missing_required(tmp_path: Path):
    with pytest.raises(TermsError, match="not found"):
        load_terms_file(tmp_path / "terms.txt", required=True)


def test_load_one_term_per_line_with_comments_and_blanks(tmp_path: Path):
    path = tmp_path / "terms.txt"
    path.write_text(
        "# names\n\nAcademind\nApp Router\n  Next.js  \n# trailing\n",
        encoding="utf-8",
    )
    context, terms = load_terms_file(path)
    assert terms == ["Academind", "App Router", "Next.js"]
    assert context == "Academind App Router Next.js"


def test_load_rejects_directory(tmp_path: Path):
    with pytest.raises(TermsError, match="not a file"):
        load_terms_file(tmp_path, required=True)


def test_terms_flag_was_explicit():
    assert terms_flag_was_explicit(["clip.wav", "--terms", "x.txt"]) is True
    assert terms_flag_was_explicit(["clip.wav", "--terms=x.txt"]) is True
    assert terms_flag_was_explicit(["clip.wav", "-o", "out"]) is False
    assert terms_flag_was_explicit(None) is False
