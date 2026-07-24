"""Load domain terms used as Qwen3-ASR context bias."""

from __future__ import annotations

from pathlib import Path

DEFAULT_TERMS_PATH = Path("terms.txt")


class TermsError(Exception):
    """Raised when a terms file cannot be loaded."""


def load_terms_file(path: Path, *, required: bool = False) -> tuple[str, list[str]]:
    """Load one term per line from ``path``.

    - Blank lines are skipped.
    - Lines whose first non-whitespace character is ``#`` are comments.
    - Each remaining line is one term (may contain spaces).
    - Terms are joined with a single space for the upstream ``context`` string.

    Returns:
        ``(context, terms)`` where ``context`` is the joined string (possibly empty).

    If the file is missing and ``required`` is False, returns ``("", [])``.
    If missing and ``required`` is True, raises ``TermsError``.
    """
    if not path.exists():
        if required:
            raise TermsError(f"terms file not found: {path}")
        return "", []
    if not path.is_file():
        raise TermsError(f"terms path is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TermsError(f"failed to read terms file {path}: {exc}") from exc

    terms: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)

    context = " ".join(terms)
    return context, terms


def terms_flag_was_explicit(argv: list[str] | None) -> bool:
    """True when the user passed ``--terms`` / ``--terms=...`` on the CLI."""
    if not argv:
        return False
    for arg in argv:
        if arg == "--terms" or arg.startswith("--terms="):
            return True
    return False
