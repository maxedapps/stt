#!/usr/bin/env python3
"""Validate real-model smoke outputs against the pinned independent oracle.

Usage:
    uv run python scripts/validate_smoke.py OUTPUT_DIR

Prints ``SMOKE OK`` on success; nonzero exit with a concise error otherwise.
Does not derive the oracle transcript from generated outputs.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Independently observed expected transcript for the pinned official sample
# (asr_en.wav). Do not replace this by reading generated artifacts.
ORACLE_TRANSCRIPT = (
    "Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening "
    "to him, but and his solo music didn't do overly well, but he did very "
    "well when he started writing for other people."
)

_SRT_TS = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$"
)
_VTT_TS = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) --> (\d{2}):(\d{2}):(\d{2})\.(\d{3})$"
)


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _ts_to_ms(h: str, m: str, s: str, ms: str) -> int:
    return int(h) * 3_600_000 + int(m) * 60_000 + int(s) * 1000 + int(ms)


def _find_artifacts(out_dir: Path) -> dict[str, Path]:
    files = [p for p in out_dir.iterdir() if p.is_file()]
    if any(".tmp-" in p.name for p in files):
        _fail("temporary files remain in output directory")
    if len(files) != 4:
        names = sorted(p.name for p in files)
        _fail(f"expected exactly 4 artifacts, found {len(files)}: {names}")

    typed: dict[str, Path] = {}
    stems: set[str] = set()
    for path in files:
        name = path.name
        if name.endswith(".words.json"):
            key, stem = "words_json", name[: -len(".words.json")]
        elif name.endswith(".txt"):
            key, stem = "txt", name[: -len(".txt")]
        elif name.endswith(".srt"):
            key, stem = "srt", name[: -len(".srt")]
        elif name.endswith(".vtt"):
            key, stem = "vtt", name[: -len(".vtt")]
        else:
            _fail(f"unexpected file in output directory: {name}")
        if key in typed:
            _fail(f"duplicate artifact kind {key!r}")
        typed[key] = path
        stems.add(stem)

    expected = {"txt", "words_json", "srt", "vtt"}
    if set(typed) != expected:
        _fail(f"artifact set mismatch: {sorted(typed)} vs {sorted(expected)}")
    if len(stems) != 1:
        _fail(f"artifact stems are not unique: {sorted(stems)}")
    return typed


def _parse_srt(text: str) -> list[tuple[int, int, str]]:
    if text == "":
        return []
    if not text.endswith("\n"):
        _fail("SRT must end with a trailing newline")
    blocks = text.split("\n\n")
    # trailing split may yield final empty from terminating blank line
    if blocks and blocks[-1] == "":
        blocks = blocks[:-1]
    cues: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 3:
            _fail(f"SRT cue {index} is malformed")
        if lines[0] != str(index):
            _fail(f"SRT cue number mismatch: expected {index}, got {lines[0]!r}")
        match = _SRT_TS.match(lines[1])
        if not match:
            _fail(f"SRT cue {index} has invalid timestamp line: {lines[1]!r}")
        start = _ts_to_ms(*match.group(1, 2, 3, 4))
        end = _ts_to_ms(*match.group(5, 6, 7, 8))
        body = "\n".join(lines[2:])
        if body.strip() == "":
            _fail(f"SRT cue {index} has empty text")
        cues.append((start, end, body))
    return cues


def _parse_vtt(text: str) -> list[tuple[int, int, str]]:
    if not text.startswith("WEBVTT\n"):
        _fail("VTT must start with WEBVTT header")
    if text == "WEBVTT\n\n":
        return []
    if not text.endswith("\n"):
        _fail("VTT must end with a trailing newline")
    rest = text[len("WEBVTT\n") :].removeprefix("\n")
    blocks = rest.split("\n\n")
    if blocks and blocks[-1] == "":
        blocks = blocks[:-1]
    cues: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            _fail(f"VTT cue {index} is malformed")
        match = _VTT_TS.match(lines[0])
        if not match:
            _fail(f"VTT cue {index} has invalid timestamp line: {lines[0]!r}")
        start = _ts_to_ms(*match.group(1, 2, 3, 4))
        end = _ts_to_ms(*match.group(5, 6, 7, 8))
        body = "\n".join(lines[1:])
        if body.strip() == "":
            _fail(f"VTT cue {index} has empty text")
        cues.append((start, end, body))
    return cues


def _validate_words(data: object) -> list[dict[str, object]]:
    if not isinstance(data, dict):
        _fail("words.json root must be an object")
    keys = list(data.keys())
    if keys != ["schema_version", "text", "language", "words"]:
        _fail(f"words.json keys must be exact schema order/set, got {keys}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        _fail("schema_version must be integer 1")
    if data["language"] != "English":
        _fail(f"language must be 'English', got {data['language']!r}")
    if not isinstance(data["text"], str):
        _fail("text must be a string")
    words = data["words"]
    if not isinstance(words, list):
        _fail("words must be a list")
    prev_end: int | None = None
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(words):
        if not isinstance(item, dict):
            _fail(f"words[{index}] must be an object")
        if list(item.keys()) != ["text", "start_ms", "end_ms"]:
            _fail(f"words[{index}] keys must be text/start_ms/end_ms")
        text = item["text"]
        start = item["start_ms"]
        end = item["end_ms"]
        if not isinstance(text, str) or text == "":
            _fail(f"words[{index}].text must be a nonempty string")
        if type(start) is not int or type(end) is not int:
            _fail(f"words[{index}] timestamps must be integers")
        if start < 0 or end < 0 or start > end:
            _fail(f"words[{index}] has invalid timestamp range")
        if prev_end is not None and prev_end > start:
            _fail(f"words[{index}] overlaps previous unit")
        prev_end = end
        normalized.append(item)
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        _fail("usage: validate_smoke.py OUTPUT_DIR")
    out_dir = Path(args[0])
    if not out_dir.is_dir():
        _fail(f"output directory not found: {out_dir}")

    artifacts = _find_artifacts(out_dir)

    txt_raw = artifacts["txt"].read_bytes()
    if not txt_raw.endswith(b"\n") or txt_raw.endswith(b"\n\n"):
        _fail("TXT must end with exactly one trailing LF")
    txt_text = txt_raw[:-1].decode("utf-8")
    if txt_text != ORACLE_TRANSCRIPT:
        _fail("TXT does not match oracle transcript")

    json_raw = artifacts["words_json"].read_bytes()
    if not json_raw.endswith(b"\n"):
        _fail("words.json must end with a trailing LF")
    try:
        data = json.loads(json_raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"words.json is not valid JSON: {exc}")
    words = _validate_words(data)
    if data["text"] != ORACLE_TRANSCRIPT:
        _fail("words.json root text does not match oracle transcript")
    if not words:
        _fail("words.json words[] is empty for nonempty oracle")

    srt_text = artifacts["srt"].read_text(encoding="utf-8")
    vtt_text = artifacts["vtt"].read_text(encoding="utf-8")
    srt_cues = _parse_srt(srt_text)
    vtt_cues = _parse_vtt(vtt_text)

    if len(srt_cues) != len(vtt_cues):
        _fail("SRT/VTT cue counts differ")
    if not srt_cues:
        _fail("expected nonempty captions for smoke fixture")

    srt_joined = _norm(" ".join(c[2] for c in srt_cues))
    vtt_joined = _norm(" ".join(c[2] for c in vtt_cues))
    oracle_norm = _norm(ORACLE_TRANSCRIPT)
    if srt_joined != oracle_norm:
        _fail("normalized SRT cue text does not equal oracle transcript")
    if vtt_joined != oracle_norm:
        _fail("normalized VTT cue text does not equal oracle transcript")

    for index, ((ss, se, st), (vs, ve, vt)) in enumerate(
        zip(srt_cues, vtt_cues, strict=True), start=1
    ):
        if (ss, se, st) != (vs, ve, vt):
            _fail(f"SRT/VTT cue {index} mismatch")
        if ss > se:
            _fail(f"cue {index} has reversed times")

    # Caption cue times should be monotonic overall.
    prev_end = -1
    for index, (start, end, _text) in enumerate(srt_cues, start=1):
        if start < prev_end:
            _fail(f"cue {index} starts before previous cue ends")
        prev_end = end

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
