"""English source-span reconciliation and caption cue generation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from stt.engine import Transcription

# Opening punctuation that pulls an interstitial gap onto the following unit.
_OPENING_CHARS = frozenset("([{“‘«‹")

# Sentence end on a unit's display text (plan T3 fixed v1).
_SENTENCE_END_RE = re.compile(r'[.!?…。！？](?:["\'”’»)\]}]+)?$')

# Cue grouping limits (fixed v1).
_MAX_UNITS = 10
_MAX_CHARS = 42
_MAX_DURATION_MS = 6000
_MAX_GAP_MS = 800


class ReconciliationError(Exception):
    """Canonical source text could not be reconciled with alignment units."""


@dataclass(frozen=True)
class SourceUnit:
    """Alignment unit with one contiguous owned canonical source range."""

    text: str
    start_ms: int
    end_ms: int
    source_start: int
    source_end: int


@dataclass(frozen=True)
class Cue:
    """One subtitle cue spanning one or more source units."""

    text: str
    start_ms: int
    end_ms: int


def clean_alignment_token(token: str) -> str:
    """Mirror upstream English aligner cleaner: keep Unicode L*/N* and ASCII '."""
    return "".join(ch for ch in token if _is_kept_aligner_char(ch))


def _is_kept_aligner_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N"))


def _collapse_ws(text: str) -> str:
    return " ".join(text.split())


def _iter_source_tokens(text: str) -> list[tuple[str, int, int]]:
    """Whitespace-split tokens with [start, end) offsets; no characters dropped."""
    tokens: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i + 1
        while j < n and not text[j].isspace():
            j += 1
        tokens.append((text[i:j], i, j))
        i = j
    return tokens


def _gap_attaches_to_following(
    source: str, gap_tokens: list[tuple[str, int, int]]
) -> bool:
    """True when interstitial gap's first non-whitespace char is an opener."""
    if not gap_tokens:
        return False
    start = gap_tokens[0][1]
    end = gap_tokens[-1][2]
    for ch in source[start:end]:
        if ch.isspace():
            continue
        return ch in _OPENING_CHARS
    return False


def restore_source_units(result: Transcription) -> tuple[SourceUnit, ...]:
    """Map each alignment unit onto one owned contiguous canonical source span.

    Fails all-or-nothing on match mismatch or unowned non-whitespace characters.
    """
    source = result.text
    units = result.units

    if source == "" and units == ():
        return ()

    tokens = _iter_source_tokens(source)
    # Classify tokens: match candidates keep cleaned form; cleaned-empty are punct gaps.
    classified: list[tuple[str, int, int, str | None]] = []
    for token, start, end in tokens:
        cleaned = clean_alignment_token(token)
        classified.append((token, start, end, cleaned if cleaned else None))

    match_indices = [i for i, item in enumerate(classified) if item[3] is not None]
    match_cleaned = [classified[i][3] for i in match_indices]
    # narrowing: match_cleaned entries are str
    assert all(c is not None for c in match_cleaned)

    if len(match_indices) != len(units):
        raise ReconciliationError(
            f"alignment unit count {len(units)} != cleaned source token count "
            f"{len(match_indices)}"
        )

    for index, (unit, cleaned) in enumerate(zip(units, match_cleaned, strict=True)):
        if unit.text != cleaned:
            raise ReconciliationError(
                f"unit[{index}] text {unit.text!r} != cleaned source token {cleaned!r}"
            )

    if not match_indices:
        # Non-whitespace remains but nothing matched (punctuation-only / no units).
        if any(not ch.isspace() for ch in source):
            raise ReconciliationError(
                "unowned non-whitespace source characters with no alignment units"
            )
        return ()

    n_tokens = len(classified)
    # For each match slot, collect token index ranges owned by that unit.
    owned_token_idx: list[list[int]] = [[] for _ in match_indices]

    def extend_owned(match_slot: int, token_i_start: int, token_i_end: int) -> None:
        owned_token_idx[match_slot].extend(range(token_i_start, token_i_end))

    first_match_i = match_indices[0]
    last_match_i = match_indices[-1]

    # Leading gap → first unit.
    if first_match_i > 0:
        extend_owned(0, 0, first_match_i)

    for slot, match_i in enumerate(match_indices):
        extend_owned(slot, match_i, match_i + 1)

        # Interstitial gap between this match and the next.
        if slot + 1 < len(match_indices):
            next_match_i = match_indices[slot + 1]
            gap_lo, gap_hi = match_i + 1, next_match_i
            if gap_lo < gap_hi:
                gap_tokens = [classified[j][:3] for j in range(gap_lo, gap_hi)]
                # All gap tokens must be punct (cleaned-empty); enforced by match list.
                if _gap_attaches_to_following(source, gap_tokens):
                    extend_owned(slot + 1, gap_lo, gap_hi)
                else:
                    extend_owned(slot, gap_lo, gap_hi)

    # Trailing gap → last unit.
    if last_match_i + 1 < n_tokens:
        extend_owned(len(match_indices) - 1, last_match_i + 1, n_tokens)

    # Build contiguous ranges; verify every non-ws char is owned exactly once.
    owned_flags = [False] * len(source)
    source_units: list[SourceUnit] = []

    for slot, unit in enumerate(units):
        idxs = owned_token_idx[slot]
        if not idxs:
            raise ReconciliationError(f"unit[{slot}] owns no source tokens")
        # Contiguous token ownership in source order.
        idxs_sorted = sorted(idxs)
        range_start = classified[idxs_sorted[0]][1]
        range_end = classified[idxs_sorted[-1]][2]
        # Mark all characters in the contiguous span as owned (incl. internal ws).
        for pos in range(range_start, range_end):
            if owned_flags[pos]:
                raise ReconciliationError(
                    f"overlapping source ownership at offset {pos}"
                )
            owned_flags[pos] = True
        display = _collapse_ws(source[range_start:range_end])
        source_units.append(
            SourceUnit(
                text=display,
                start_ms=unit.start_ms,
                end_ms=unit.end_ms,
                source_start=range_start,
                source_end=range_end,
            )
        )

    for pos, ch in enumerate(source):
        if not ch.isspace() and not owned_flags[pos]:
            raise ReconciliationError(
                f"unowned non-whitespace source character {ch!r} at offset {pos}"
            )

    return tuple(source_units)


def group_cues(
    units: tuple[SourceUnit, ...] | list[SourceUnit], source_text: str
) -> list[Cue]:
    """Group restored source units into fixed-v1 subtitle cues."""
    if not units:
        return []

    cues: list[Cue] = []
    current: list[SourceUnit] = []

    def cue_text_for(group: list[SourceUnit]) -> str:
        start = group[0].source_start
        end = group[-1].source_end
        return _collapse_ws(source_text[start:end])

    def flush() -> None:
        nonlocal current
        if not current:
            return
        cues.append(
            Cue(
                text=cue_text_for(current),
                start_ms=current[0].start_ms,
                end_ms=current[-1].end_ms,
            )
        )
        current = []

    for unit in units:
        if not current:
            current = [unit]
            continue

        gap = unit.start_ms - current[-1].end_ms
        duration = unit.end_ms - current[0].start_ms
        next_group = [*current, unit]
        next_text = cue_text_for(next_group)
        current_text = cue_text_for(current)

        should_break = False
        if gap >= _MAX_GAP_MS:
            should_break = True
        if duration > _MAX_DURATION_MS:
            should_break = True
        if len(current) >= _MAX_UNITS:
            should_break = True
        if len(next_text) > _MAX_CHARS and len(current_text) > 0:
            should_break = True
        if _SENTENCE_END_RE.search(current[-1].text):
            should_break = True

        if should_break:
            flush()
            current = [unit]
        else:
            current.append(unit)

    flush()
    return cues


def _format_timestamp(ms: int, *, fractional_sep: str) -> str:
    total_ms = max(0, int(ms))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{fractional_sep}{millis:03d}"


def render_srt(cues: list[Cue] | tuple[Cue, ...]) -> str:
    """Render SRT text from cues (empty cue list → empty string)."""
    if not cues:
        return ""
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        start = _format_timestamp(cue.start_ms, fractional_sep=",")
        end = _format_timestamp(cue.end_ms, fractional_sep=",")
        blocks.append(f"{index}\n{start} --> {end}\n{cue.text}\n")
    return "\n".join(blocks) + "\n"


def render_vtt(cues: list[Cue] | tuple[Cue, ...]) -> str:
    """Render WebVTT text from cues (empty → valid header-only VTT)."""
    if not cues:
        return "WEBVTT\n\n"
    blocks: list[str] = ["WEBVTT\n"]
    for cue in cues:
        start = _format_timestamp(cue.start_ms, fractional_sep=".")
        end = _format_timestamp(cue.end_ms, fractional_sep=".")
        blocks.append(f"{start} --> {end}\n{cue.text}\n")
    return "\n".join(blocks) + "\n"
