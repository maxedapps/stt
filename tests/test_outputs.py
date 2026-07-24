"""Artifact bytes, schema, collision races, and atomic publish tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stt.captions import restore_source_units
from stt.engine import AlignmentUnit, Transcription
from stt.outputs import (
    OutputCollisionError,
    PublishError,
    preflight_targets,
    publish_artifacts,
    render_payloads,
    render_txt,
    render_words_json,
    target_paths,
)


def _unit(text: str, start_ms: int, end_ms: int) -> AlignmentUnit:
    return AlignmentUnit(text=text, start_ms=start_ms, end_ms=end_ms)


def _result(
    text: str,
    unit_texts: list[str] | None = None,
    *,
    t0: int = 0,
    dur: int = 100,
) -> Transcription:
    if unit_texts is None:
        unit_texts = []
    units: list[AlignmentUnit] = []
    t = t0
    for ut in unit_texts:
        units.append(_unit(ut, t, t + dur))
        t += dur
    return Transcription(text=text, language="English", units=tuple(units))


def _nonempty() -> Transcription:
    return _result("Hello, world!", ["Hello", "world"])


def _empty() -> Transcription:
    return _result("", [])


def _temps(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and ".tmp-" in p.name)


# ---------------------------------------------------------------------------
# path helpers / preflight
# ---------------------------------------------------------------------------


def test_target_paths_from_stem(tmp_path: Path):
    media = tmp_path / "clip.wav"
    out = tmp_path / "out"
    paths = target_paths(media, out)
    assert paths == {
        "txt": out / "clip.txt",
        "words_json": out / "clip.words.json",
        "srt": out / "clip.srt",
        "vtt": out / "clip.vtt",
    }


def test_preflight_rejects_existing_without_overwrite(tmp_path: Path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    out = tmp_path
    (out / "clip.txt").write_text("old\n", encoding="utf-8")
    with pytest.raises(OutputCollisionError, match="already exists"):
        preflight_targets(media, target_paths(media, out), overwrite=False)


def test_preflight_allows_existing_with_overwrite(tmp_path: Path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    out = tmp_path
    (out / "clip.txt").write_text("old\n", encoding="utf-8")
    preflight_targets(media, target_paths(media, out), overwrite=True)


def test_preflight_rejects_directory_target(tmp_path: Path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"x")
    out = tmp_path
    (out / "clip.srt").mkdir()
    with pytest.raises(OutputCollisionError, match="directory"):
        preflight_targets(media, target_paths(media, out), overwrite=True)


def test_preflight_rejects_lexical_alias_even_with_overwrite(tmp_path: Path):
    # Input stem yields a .txt target that is the input itself.
    media = tmp_path / "note.txt"
    media.write_text("audio-shaped\n", encoding="utf-8")
    with pytest.raises(OutputCollisionError, match="aliases input"):
        preflight_targets(media, target_paths(media, tmp_path), overwrite=True)


def test_preflight_rejects_samefile_alias_even_with_overwrite(tmp_path: Path):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"audio")
    alias = tmp_path / "talk.txt"
    os.link(media, alias)
    with pytest.raises(OutputCollisionError, match="aliases input"):
        preflight_targets(media, target_paths(media, tmp_path), overwrite=True)


def test_preflight_rejects_resolve_alias(tmp_path: Path):
    media = tmp_path / "nested" / "talk.wav"
    media.parent.mkdir()
    media.write_bytes(b"audio")
    # output_dir resolves such that talk.txt == media when stem conflicts via ..
    # Use input path that resolves to a target name.
    linked_dir = tmp_path / "outlink"
    linked_dir.symlink_to(tmp_path / "nested", target_is_directory=True)
    # media stem talk → talk.txt in nested/; also create as target via outlink
    target = linked_dir / "talk.txt"
    os.link(media, target)
    with pytest.raises(OutputCollisionError, match="aliases input"):
        preflight_targets(media, target_paths(media, linked_dir), overwrite=True)


# ---------------------------------------------------------------------------
# exact bytes
# ---------------------------------------------------------------------------


def test_empty_result_exact_bytes():
    payloads = render_payloads(_empty())
    assert payloads["txt"] == b"\n"
    assert payloads["srt"] == b""
    assert payloads["vtt"] == b"WEBVTT\n\n"
    expected_json = (
        b'{\n  "schema_version": 1,\n  "text": "",\n  "language": "English",\n  '
        b'"words": []\n}\n'
    )
    assert payloads["words_json"] == expected_json


def test_nonempty_txt_exact_bytes():
    result = _nonempty()
    assert render_txt(result) == b"Hello, world!\n"


def test_nonempty_words_json_schema_and_bytes():
    result = _nonempty()
    units = restore_source_units(result)
    raw = render_words_json(result, units)
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")
    data = json.loads(text)
    assert list(data.keys()) == ["schema_version", "text", "language", "words"]
    assert data["schema_version"] == 1
    assert data["text"] == "Hello, world!"
    assert data["language"] == "English"
    assert data["words"] == [
        {"text": "Hello,", "start_ms": 0, "end_ms": 100},
        {"text": "world!", "start_ms": 100, "end_ms": 200},
    ]
    # ensure_ascii=False + indent 2 formatting
    assert raw == (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()


def test_nonempty_words_preserve_unicode_punctuation():
    result = _result("It’s “state-of-the-art”.", ["Its", "stateoftheart"])
    units = restore_source_units(result)
    data = json.loads(render_words_json(result, units).decode())
    assert data["text"] == "It’s “state-of-the-art”."
    assert data["words"][0]["text"] == "It’s"
    assert "“" in data["words"][1]["text"] and "”" in data["words"][1]["text"]
    # UTF-8, not escaped ascii
    raw = render_words_json(result, units)
    assert "\\u" not in raw.decode("utf-8")


def test_nonempty_srt_vtt_punctuated():
    payloads = render_payloads(_nonempty())
    srt = payloads["srt"].decode()
    vtt = payloads["vtt"].decode()
    assert "Hello, world!" in srt
    assert "Hello, world!" in vtt
    assert srt.startswith("1\n")
    assert "-->" in srt and "," in srt.split("-->")[0]
    assert vtt.startswith("WEBVTT\n")
    assert "." in vtt.split("-->")[0] or vtt.count("-->") >= 1


# ---------------------------------------------------------------------------
# publish success paths
# ---------------------------------------------------------------------------


def test_publish_empty_writes_exact_files(tmp_path: Path):
    media = tmp_path / "silence.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    paths = publish_artifacts(media, out, _empty(), overwrite=False)
    assert [p.name for p in paths] == [
        "silence.txt",
        "silence.words.json",
        "silence.srt",
        "silence.vtt",
    ]
    assert (out / "silence.txt").read_bytes() == b"\n"
    assert (out / "silence.srt").read_bytes() == b""
    assert (out / "silence.vtt").read_bytes() == b"WEBVTT\n\n"
    data = json.loads((out / "silence.words.json").read_text(encoding="utf-8"))
    assert data == {
        "schema_version": 1,
        "text": "",
        "language": "English",
        "words": [],
    }
    assert _temps(out) == []


def test_publish_nonempty_and_overwrite(tmp_path: Path):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    first = publish_artifacts(media, out, _nonempty(), overwrite=False)
    assert all(p.is_file() for p in first)
    txt_before = (out / "talk.txt").read_bytes()
    assert txt_before == b"Hello, world!\n"

    other = _result("Go!", ["Go"])
    second = publish_artifacts(media, out, other, overwrite=True)
    assert [p.name for p in second] == [
        "talk.txt",
        "talk.words.json",
        "talk.srt",
        "talk.vtt",
    ]
    assert (out / "talk.txt").read_bytes() == b"Go!\n"
    assert _temps(out) == []


def test_publish_no_overwrite_collision(tmp_path: Path):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    publish_artifacts(media, out, _nonempty(), overwrite=False)
    with pytest.raises(OutputCollisionError, match="already exists"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)
    assert (out / "talk.txt").read_bytes() == b"Hello, world!\n"
    assert _temps(out) == []


def test_publish_refuses_input_alias_with_overwrite(tmp_path: Path):
    media = tmp_path / "note.txt"
    media.write_text("keep-me\n", encoding="utf-8")
    with pytest.raises(OutputCollisionError, match="aliases input"):
        publish_artifacts(media, tmp_path, _nonempty(), overwrite=True)
    assert media.read_text(encoding="utf-8") == "keep-me\n"
    assert _temps(tmp_path) == []


def test_publish_refuses_samefile_alias_with_overwrite(tmp_path: Path):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"audio-bytes")
    os.link(media, tmp_path / "talk.txt")
    with pytest.raises(OutputCollisionError, match="aliases input"):
        publish_artifacts(media, tmp_path, _nonempty(), overwrite=True)
    assert media.read_bytes() == b"audio-bytes"
    assert (tmp_path / "talk.txt").read_bytes() == b"audio-bytes"
    assert _temps(tmp_path) == []


# ---------------------------------------------------------------------------
# races / failure injection
# ---------------------------------------------------------------------------


def test_race_destination_after_preflight_preserves_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    racing = out / "talk.txt"
    racer_bytes = b"racer-content\n"

    real_commit = __import__("stt.outputs", fromlist=["_commit_one"])._commit_one

    def racing_commit(temp: Path, target: Path, *, overwrite: bool) -> None:
        if target == racing and not target.exists():
            target.write_bytes(racer_bytes)
        real_commit(temp, target, overwrite=overwrite)

    monkeypatch.setattr("stt.outputs._commit_one", racing_commit)

    with pytest.raises(OutputCollisionError, match="already exists"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)

    assert racing.read_bytes() == racer_bytes
    # First commit failed; no other artifacts should exist (txt is first).
    assert not (out / "talk.words.json").exists()
    assert not (out / "talk.srt").exists()
    assert not (out / "talk.vtt").exists()
    assert _temps(out) == []


def test_race_on_later_artifact_leaves_complete_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    racing = out / "talk.srt"
    racer_bytes = b"partial-race\n"

    real_commit = __import__("stt.outputs", fromlist=["_commit_one"])._commit_one

    def racing_commit(temp: Path, target: Path, *, overwrite: bool) -> None:
        if target == racing and not target.exists():
            target.write_bytes(racer_bytes)
        real_commit(temp, target, overwrite=overwrite)

    monkeypatch.setattr("stt.outputs._commit_one", racing_commit)

    with pytest.raises(OutputCollisionError, match="already exists"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)

    assert racing.read_bytes() == racer_bytes
    # Earlier commits are complete artifacts.
    assert (out / "talk.txt").read_bytes() == b"Hello, world!\n"
    words = json.loads((out / "talk.words.json").read_text(encoding="utf-8"))
    assert words["schema_version"] == 1
    assert words["text"] == "Hello, world!"
    assert not (out / "talk.vtt").exists()
    assert _temps(out) == []


def test_write_failure_cleans_temps_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()

    calls = {"n": 0}
    real_write = __import__("stt.outputs", fromlist=["_write_bytes"])._write_bytes

    def flaky_write(path: Path, data: bytes) -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise OSError("disk full")
        real_write(path, data)

    monkeypatch.setattr("stt.outputs._write_bytes", flaky_write)

    with pytest.raises(PublishError, match="temporary artifact"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)

    assert list(out.iterdir()) == []
    assert _temps(out) == []


def test_commit_failure_cleans_remaining_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()

    real_commit = __import__("stt.outputs", fromlist=["_commit_one"])._commit_one
    state = {"n": 0}

    def flaky_commit(temp: Path, target: Path, *, overwrite: bool) -> None:
        state["n"] += 1
        if state["n"] == 2:
            raise OSError("quota exceeded")
        real_commit(temp, target, overwrite=overwrite)

    monkeypatch.setattr("stt.outputs._commit_one", flaky_commit)

    with pytest.raises(PublishError, match="failed to commit"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)

    # First artifact committed completely.
    assert (out / "talk.txt").read_bytes() == b"Hello, world!\n"
    assert not (out / "talk.words.json").exists()
    assert _temps(out) == []


def test_overwrite_replace_failure_preserves_existing_and_cleans_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()
    publish_artifacts(media, out, _nonempty(), overwrite=False)
    existing = {
        p.name: p.read_bytes()
        for p in out.iterdir()
        if p.is_file() and ".tmp-" not in p.name
    }

    def boom_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", boom_replace)

    with pytest.raises(PublishError, match="failed to commit"):
        publish_artifacts(media, out, _result("Go!", ["Go"]), overwrite=True)

    after = {
        p.name: p.read_bytes()
        for p in out.iterdir()
        if p.is_file() and ".tmp-" not in p.name
    }
    assert after == existing
    assert _temps(out) == []


def test_link_failure_other_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"x")
    out = tmp_path / "out"
    out.mkdir()

    def boom_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        raise OSError("link not supported")

    monkeypatch.setattr(os, "link", boom_link)

    with pytest.raises(PublishError, match="failed to commit"):
        publish_artifacts(media, out, _nonempty(), overwrite=False)

    assert [p for p in out.iterdir() if ".tmp-" not in p.name] == []
    assert _temps(out) == []
