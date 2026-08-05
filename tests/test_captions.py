"""Model-free caption reconciliation, grouping, and render tests."""

from __future__ import annotations

import re
from itertools import pairwise

import pytest

from stt.captions import (
    Cue,
    ReconciliationError,
    SourceUnit,
    clean_alignment_token,
    group_cues,
    render_srt,
    render_vtt,
    restore_source_units,
)
from stt.engine import AlignmentUnit, Transcription

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split())


def _unit(text: str, start_ms: int, end_ms: int) -> AlignmentUnit:
    return AlignmentUnit(text=text, start_ms=start_ms, end_ms=end_ms)


def _result(
    text: str, unit_texts: list[str], *, t0: int = 0, dur: int = 100
) -> Transcription:
    """Build a Transcription with sequential non-overlapping unit times."""
    units: list[AlignmentUnit] = []
    t = t0
    for ut in unit_texts:
        units.append(_unit(ut, t, t + dur))
        t += dur
    return Transcription(text=text, language="English", units=tuple(units))


def _restore(text: str, unit_texts: list[str], **kwargs: int) -> tuple[SourceUnit, ...]:
    return restore_source_units(_result(text, unit_texts, **kwargs))


def _owned_slices(source: str, units: tuple[SourceUnit, ...]) -> list[str]:
    return [source[u.source_start : u.source_end] for u in units]


def _all_owned_non_ws(source: str, units: tuple[SourceUnit, ...]) -> str:
    """Concatenate owned ranges and drop whitespace — every non-ws once."""
    chars: list[str] = []
    for u in units:
        chars.append(source[u.source_start : u.source_end])
    return "".join(ch for ch in "".join(chars) if not ch.isspace())


def _source_non_ws(source: str) -> str:
    return "".join(ch for ch in source if not ch.isspace())


# ---------------------------------------------------------------------------
# clean_alignment_token
# ---------------------------------------------------------------------------


def test_clean_keeps_letters_numbers_ascii_apostrophe():
    assert clean_alignment_token("Hello") == "Hello"
    assert clean_alignment_token("3.14") == "314"
    assert clean_alignment_token("can't") == "can't"
    assert clean_alignment_token("state-of-the-art") == "stateoftheart"
    assert clean_alignment_token("can't—really—stop.") == "can'treallystop"


def test_clean_deletes_curly_apostrophe_and_other_punct():
    assert clean_alignment_token("It’s") == "Its"  # U+2019
    assert clean_alignment_token('"Go,') == "Go"
    assert clean_alignment_token("world!") == "world"
    assert clean_alignment_token("…") == ""
    assert clean_alignment_token("—") == ""
    assert clean_alignment_token("(") == ""


def test_clean_keeps_unicode_letters_and_digits():
    assert clean_alignment_token("café") == "café"
    assert clean_alignment_token("Αθήνα") == "Αθήνα"
    assert clean_alignment_token("１２３") == "１２３"  # fullwidth digits are N*


# ---------------------------------------------------------------------------
# restore_source_units — exact restoration
# ---------------------------------------------------------------------------


def test_restore_hello_world():
    source = "Hello, world!"
    units = _restore(source, ["Hello", "world"])
    assert len(units) == 2
    assert units[0].text == "Hello,"
    assert units[1].text == "world!"
    assert _norm(" ".join(u.text for u in units)) == _norm(source)
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_restore_cant_really_stop_one_unit():
    source = "I can't—really—stop."
    units = _restore(source, ["I", "can'treallystop"])
    assert [u.text for u in units] == ["I", "can't—really—stop."]
    assert units[0].start_ms == 0
    assert units[1].start_ms == 100


def test_restore_curly_apostrophe_state_of_the_art():
    source = "It’s state-of-the-art."
    units = _restore(source, ["Its", "stateoftheart"])
    assert units[0].text == "It’s"
    assert units[1].text == "state-of-the-art."


def test_restore_quoted_go_go():
    source = '"Go, go!"'
    units = _restore(source, ["Go", "go"])
    assert units[0].text == '"Go,'
    assert units[1].text == 'go!"'
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_restore_decimal_number():
    source = "3.14"
    units = _restore(source, ["314"])
    assert len(units) == 1
    assert units[0].text == "3.14"


def test_restore_repeated_words():
    source = "go go go!"
    units = _restore(source, ["go", "go", "go"])
    assert [u.text for u in units] == ["go", "go", "go!"]
    # Order preserved; timestamps progressive.
    assert [u.start_ms for u in units] == [0, 100, 200]


def test_restore_unicode_letters():
    source = "café naïve"
    units = _restore(source, ["café", "naïve"])
    assert [u.text for u in units] == ["café", "naïve"]


def test_restore_standalone_leading_interstitial_trailing_punct_tokens():
    source = "! Hello ... world ?"
    units = _restore(source, ["Hello", "world"])
    assert units[0].text == "! Hello ..."
    assert units[1].text == "world ?"
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


# ---------------------------------------------------------------------------
# restore_source_units — ownership policy
# ---------------------------------------------------------------------------


def test_ownership_curly_quoted_hello():
    # Leading/trailing curly quotes and bang as separate tokens.
    source = "“ Hello ! ”"
    units = _restore(source, ["Hello"])
    assert len(units) == 1
    assert units[0].text == _norm(source)
    assert source[units[0].source_start : units[0].source_end] == source
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_ownership_opening_paren_attaches_to_following():
    source = "Hello ( — world"
    units = _restore(source, ["Hello", "world"])
    slices = _owned_slices(source, units)
    assert _norm(slices[0]) == "Hello"
    assert _norm(slices[1]) == "( — world"
    # Gap must not appear on both sides.
    assert "(" not in slices[0]
    assert "—" not in slices[0]
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_ownership_em_dash_attaches_to_preceding():
    source = "Hello — world"
    units = _restore(source, ["Hello", "world"])
    slices = _owned_slices(source, units)
    assert _norm(slices[0]) == "Hello —"
    assert _norm(slices[1]) == "world"
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_ownership_leading_exclamation():
    source = "! Hello"
    units = _restore(source, ["Hello"])
    assert units[0].text == "! Hello"
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_ownership_trailing_exclamation_token():
    source = "Hello !"
    units = _restore(source, ["Hello"])
    assert units[0].text == "Hello !"


def test_ownership_ellipses_em_dashes_slashes():
    source = "Hello … world — ok / fine"
    units = _restore(source, ["Hello", "world", "ok", "fine"])
    slices = [_norm(s) for s in _owned_slices(source, units)]
    # Interstitial non-openers attach to preceding.
    assert slices[0] == "Hello …"
    assert slices[1] == "world —"
    assert slices[2] == "ok /"
    assert slices[3] == "fine"
    assert _all_owned_non_ws(source, units) == _source_non_ws(source)


def test_ownership_standalone_ascii_apostrophe_is_own_unit():
    source = "rock ' n ' roll"
    units = _restore(source, ["rock", "'", "n", "'", "roll"])
    assert [u.text for u in units] == ["rock", "'", "n", "'", "roll"]
    assert all(u.text for u in units)


def test_ownership_each_punct_char_exactly_once():
    source = "“ Hello ( world ) ! ”"
    units = _restore(source, ["Hello", "world"])
    # Reconstruct non-ws from owned ranges — must equal source non-ws, no dups.
    owned = _all_owned_non_ws(source, units)
    assert owned == _source_non_ws(source)
    # Ranges must not overlap.
    spans = sorted((u.source_start, u.source_end) for u in units)
    for (_a0, a1), (b0, _b1) in pairwise(spans):
        assert a1 <= b0


def test_ownership_bracket_gap_to_following_paren_to_following():
    source = "Hello [ world"
    units = _restore(source, ["Hello", "world"])
    assert _norm(_owned_slices(source, units)[0]) == "Hello"
    assert _norm(_owned_slices(source, units)[1]) == "[ world"


def test_ownership_closing_paren_to_preceding():
    source = "Hello world )"
    units = _restore(source, ["Hello", "world"])
    assert _norm(_owned_slices(source, units)[0]) == "Hello"
    assert _norm(_owned_slices(source, units)[1]) == "world )"


def test_fail_punctuation_with_no_neighboring_unit():
    with pytest.raises(ReconciliationError):
        _restore("!!!", [])


def test_fail_punctuation_only_with_empty_units_tuple():
    result = Transcription(text="...", language="English", units=())
    with pytest.raises(ReconciliationError):
        restore_source_units(result)


# ---------------------------------------------------------------------------
# restore_source_units — segment integrity / mismatch
# ---------------------------------------------------------------------------


def test_timestamps_and_count_preserved():
    result = Transcription(
        text="one two three",
        language="English",
        units=(
            _unit("one", 10, 20),
            _unit("two", 20, 35),
            _unit("three", 40, 50),
        ),
    )
    units = restore_source_units(result)
    assert len(units) == 3
    assert [(u.start_ms, u.end_ms) for u in units] == [(10, 20), (20, 35), (40, 50)]
    assert [u.text for u in units] == ["one", "two", "three"]


def test_fail_missing_unit():
    with pytest.raises(ReconciliationError):
        _restore("one two three", ["one", "two"])


def test_fail_extra_unit():
    with pytest.raises(ReconciliationError):
        _restore("one two", ["one", "two", "three"])


def test_fail_reordered_units():
    with pytest.raises(ReconciliationError):
        _restore("one two three", ["one", "three", "two"])


def test_fail_aligned_prefix_only():
    with pytest.raises(ReconciliationError):
        _restore("Hello, world!", ["Hello"])


def test_fail_content_mismatch_same_count():
    with pytest.raises(ReconciliationError):
        _restore("Hello world", ["Hello", "planet"])


def test_fail_case_sensitive_no_folding():
    with pytest.raises(ReconciliationError):
        _restore("Hello world", ["hello", "world"])


def test_empty_transcription():
    result = Transcription(text="", language="English", units=())
    assert restore_source_units(result) == ()


# ---------------------------------------------------------------------------
# group_cues — boundaries
# ---------------------------------------------------------------------------


def _timed_units(
    texts: list[str],
    *,
    start: int = 0,
    unit_dur: int = 100,
    gap: int = 0,
) -> tuple[tuple[SourceUnit, ...], str]:
    """Build SourceUnits with known source offsets over a joined transcript."""
    source_parts: list[str] = []
    units: list[SourceUnit] = []
    t = start
    offset = 0
    for i, text in enumerate(texts):
        if i > 0:
            source_parts.append(" ")
            offset += 1
        source_parts.append(text)
        end_off = offset + len(text)
        units.append(
            SourceUnit(
                text=text,
                start_ms=t,
                end_ms=t + unit_dur,
                source_start=offset,
                source_end=end_off,
            )
        )
        offset = end_off
        t = t + unit_dur + gap
    source = "".join(source_parts)
    return tuple(units), source


def test_group_gives_collapsed_alignment_a_positive_caption_interval():
    unit = SourceUnit(
        text="Okay.",
        start_ms=69_875,
        end_ms=69_875,
        source_start=0,
        source_end=5,
    )

    cues = group_cues((unit,), "Okay.")

    assert cues == [Cue(text="Okay.", start_ms=69_875, end_ms=69_876)]


def test_group_max_10_units_allows_10():
    texts = [f"w{i}" for i in range(10)]
    units, source = _timed_units(texts, unit_dur=50, gap=0)
    cues = group_cues(units, source)
    assert len(cues) == 1
    assert cues[0].text == _norm(source)


def test_group_max_10_units_breaks_at_11():
    texts = [f"w{i}" for i in range(11)]
    units, source = _timed_units(texts, unit_dur=50, gap=0)
    cues = group_cues(units, source)
    assert len(cues) == 2
    # First cue consumes 10 units.
    first_unit_count = 10
    assert cues[0].start_ms == units[0].start_ms
    assert cues[0].end_ms == units[first_unit_count - 1].end_ms
    assert cues[1].start_ms == units[10].start_ms


def test_group_max_42_chars_allows_42():
    # Two units whose combined collapsed text is exactly 42 chars.
    a = "a" * 20
    b = "b" * 21  # 20 + 1 space + 21 = 42
    units, source = _timed_units([a, b], unit_dur=100, gap=0)
    assert len(_norm(source)) == 42
    cues = group_cues(units, source)
    assert len(cues) == 1
    assert len(cues[0].text) == 42


def test_group_max_42_chars_breaks_at_43():
    a = "a" * 20
    b = "b" * 22  # 20 + 1 + 22 = 43
    units, source = _timed_units([a, b], unit_dur=100, gap=0)
    assert len(_norm(source)) == 43
    cues = group_cues(units, source)
    assert len(cues) == 2
    assert cues[0].text == a
    assert cues[1].text == b


def test_group_max_6000_ms_allows_6000():
    # duration = last.end - first.start; equal to 6000 is allowed (break on >).
    # Keep gap < 800 so only the duration limit is under test.
    u0 = SourceUnit(text="aa", start_ms=0, end_ms=100, source_start=0, source_end=2)
    u1 = SourceUnit(text="bb", start_ms=100, end_ms=6000, source_start=3, source_end=5)
    source = "aa bb"
    cues = group_cues((u0, u1), source)
    assert len(cues) == 1
    assert cues[0].end_ms - cues[0].start_ms == 6000


def test_group_max_6000_ms_breaks_at_6001():
    u0 = SourceUnit(text="aa", start_ms=0, end_ms=100, source_start=0, source_end=2)
    u1 = SourceUnit(text="bb", start_ms=100, end_ms=6001, source_start=3, source_end=5)
    source = "aa bb"
    cues = group_cues((u0, u1), source)
    assert len(cues) == 2


def test_group_gap_799_no_break():
    units, source = _timed_units(["aa", "bb"], unit_dur=100, gap=799)
    # gap between end of first and start of second = 799
    assert units[1].start_ms - units[0].end_ms == 799
    cues = group_cues(units, source)
    assert len(cues) == 1


def test_group_gap_800_breaks():
    units, source = _timed_units(["aa", "bb"], unit_dur=100, gap=800)
    assert units[1].start_ms - units[0].end_ms == 800
    cues = group_cues(units, source)
    assert len(cues) == 2


def test_group_sentence_end_before_closing_quotes():
    u0 = SourceUnit(
        text='Hello."',
        start_ms=0,
        end_ms=100,
        source_start=0,
        source_end=7,
    )
    u1 = SourceUnit(
        text="Next",
        start_ms=100,
        end_ms=200,
        source_start=8,
        source_end=12,
    )
    source = 'Hello." Next'
    assert re.search(r'[.!?…。！？](?:["\'”’»)\]}]+)?$', u0.text)
    cues = group_cues((u0, u1), source)
    assert len(cues) == 2
    assert cues[0].text == 'Hello."'
    assert cues[1].text == "Next"


def test_group_sentence_end_plain_period():
    units, source = _timed_units(["Hi.", "There"], unit_dur=100, gap=0)
    cues = group_cues(units, source)
    assert len(cues) == 2


def test_group_empty_units():
    assert group_cues((), "") == []
    assert group_cues((), "unused") == []


def test_group_cue_timing_spans_first_start_last_end():
    units, source = _timed_units(["a", "b", "c"], unit_dur=50, gap=10)
    cues = group_cues(units, source)
    assert len(cues) == 1
    assert cues[0].start_ms == units[0].start_ms
    assert cues[0].end_ms == units[-1].end_ms


def test_group_cue_text_from_canonical_span_not_aligner_join():
    # Restored units include punctuation; cue text uses source span.
    source = "Hello, world!"
    restored = _restore(source, ["Hello", "world"])
    cues = group_cues(restored, source)
    assert len(cues) == 1
    assert cues[0].text == "Hello, world!"


# ---------------------------------------------------------------------------
# render_srt / render_vtt
# ---------------------------------------------------------------------------


def test_render_empty_cues():
    assert render_srt([]) == ""
    assert render_vtt([]) == "WEBVTT\n\n"


def test_render_timestamp_second_minute_hour_carries():
    cues = [
        Cue(text="a", start_ms=0, end_ms=1_000),  # 1s
        Cue(text="b", start_ms=59_000, end_ms=60_000),  # minute boundary
        Cue(text="c", start_ms=3_599_000, end_ms=3_600_001),  # hour boundary
    ]
    srt = render_srt(cues)
    assert "00:00:00,000 --> 00:00:01,000" in srt
    assert "00:00:59,000 --> 00:01:00,000" in srt
    assert "00:59:59,000 --> 01:00:00,001" in srt

    vtt = render_vtt(cues)
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:01.000" in vtt
    assert "00:00:59.000 --> 00:01:00.000" in vtt
    assert "00:59:59.000 --> 01:00:00.001" in vtt


def test_render_srt_numbered_and_structure():
    cues = [
        Cue(text="Hello, world!", start_ms=0, end_ms=500),
        Cue(text="Next line", start_ms=600, end_ms=900),
    ]
    srt = render_srt(cues)
    assert srt == (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "Hello, world!\n"
        "\n"
        "2\n"
        "00:00:00,600 --> 00:00:00,900\n"
        "Next line\n"
        "\n"
    )


def test_render_vtt_structure():
    cues = [Cue(text="Hello, world!", start_ms=0, end_ms=500)]
    vtt = render_vtt(cues)
    assert vtt == ("WEBVTT\n\n00:00:00.000 --> 00:00:00.500\nHello, world!\n\n")


def test_srt_vtt_cue_texts_identical_and_match_canonical():
    source = 'Hello, world! It’s state-of-the-art. "Go, go!"'
    unit_texts = ["Hello", "world", "Its", "stateoftheart", "Go", "go"]
    # Build realistic sequential times.
    result = _result(source, unit_texts, dur=200)
    restored = restore_source_units(result)
    cues = group_cues(restored, source)

    srt = render_srt(cues)
    vtt = render_vtt(cues)

    # Extract cue texts from both formats.
    srt_texts = re.findall(
        r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+?)\n",
        srt,
    )
    vtt_texts = re.findall(
        r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n(.+?)\n",
        vtt,
    )
    assert srt_texts == vtt_texts
    assert [_norm(t) for t in srt_texts] == [_norm(c.text) for c in cues]

    concatenated = _norm(" ".join(c.text for c in cues))
    assert concatenated == _norm(source)


def test_end_to_end_punctuation_preserved_in_cues():
    source = "I can't—really—stop."
    restored = _restore(source, ["I", "can'treallystop"])
    cues = group_cues(restored, source)
    assert len(cues) == 1
    assert cues[0].text == source
    assert "can't—really—stop" in render_srt(cues)
    assert "can't—really—stop" in render_vtt(cues)
