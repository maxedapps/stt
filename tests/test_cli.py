"""CLI contract tests (model-free)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from stt.cli import run


def test_help_exit_zero_and_contract():
    result = subprocess.run(
        [sys.executable, "-m", "stt.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    out = result.stdout
    assert "INPUT" in out
    assert "--output-dir" in out
    assert "--overwrite" in out
    assert "--terms" in out
    # No unsupported feature flags
    lowered = out.lower()
    for banned in (
        "--language",
        "--model",
        "--format",
        "--stream",
        "--diariz",
        "--batch",
        "--device",
        "--quantize",
    ):
        assert banned not in lowered


def test_missing_input(tmp_path: Path):
    missing = tmp_path / "nope.wav"
    code = run([str(missing), "-o", str(tmp_path / "out")])
    assert code == 1


def test_directory_as_input_rejected(tmp_path: Path):
    code = run([str(tmp_path), "-o", str(tmp_path / "out")])
    assert code == 1


def test_nested_output_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    nested = tmp_path / "a" / "b" / "c"
    published: list[Path] = []

    def fake_transcribe(
        input_path: Path, output_dir: Path, *, overwrite: bool, context: str = ""
    ):
        assert input_path == media
        assert output_dir == nested
        assert nested.is_dir()
        assert overwrite is False
        assert context == ""
        paths = [
            nested / "clip.txt",
            nested / "clip.words.json",
            nested / "clip.srt",
            nested / "clip.vtt",
        ]
        for p in paths:
            p.write_text("ok\n", encoding="utf-8")
        published.extend(paths)
        return paths

    monkeypatch.setattr("stt.cli.transcribe_file", fake_transcribe)
    code = run([str(media), "-o", str(nested)])
    assert code == 0
    assert nested.is_dir()
    assert len(published) == 4


def test_output_path_as_file_rejected(tmp_path: Path):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    not_dir = tmp_path / "file-not-dir"
    not_dir.write_text("x", encoding="utf-8")
    code = run([str(media), "-o", str(not_dir)])
    assert code == 1


def test_mocked_success_without_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"

    def fake_transcribe(
        input_path: Path, output_dir: Path, *, overwrite: bool, context: str = ""
    ):
        assert overwrite is True
        assert context == ""
        paths = [
            output_dir / "talk.txt",
            output_dir / "talk.words.json",
            output_dir / "talk.srt",
            output_dir / "talk.vtt",
        ]
        for p in paths:
            p.write_text("ok\n", encoding="utf-8")
        return paths

    monkeypatch.setattr("stt.cli.transcribe_file", fake_transcribe)
    code = run([str(media), "-o", str(out), "--overwrite"])
    assert code == 0
    captured = capsys.readouterr()
    lines = [ln for ln in captured.out.splitlines() if ln.strip()]
    assert len(lines) == 4
    for line in lines:
        assert Path(line).exists()


def test_default_terms_file_optional_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"
    monkeypatch.chdir(tmp_path)

    def fake_transcribe(
        input_path: Path, output_dir: Path, *, overwrite: bool, context: str = ""
    ):
        assert context == ""
        paths = [
            output_dir / "talk.txt",
            output_dir / "talk.words.json",
            output_dir / "talk.srt",
            output_dir / "talk.vtt",
        ]
        for p in paths:
            p.write_text("ok\n", encoding="utf-8")
        return paths

    monkeypatch.setattr("stt.cli.transcribe_file", fake_transcribe)
    code = run([str(media), "-o", str(out)])
    assert code == 0
    err = capsys.readouterr().err
    assert "no terms file at terms.txt" in err


def test_default_terms_file_loaded_from_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"
    (tmp_path / "terms.txt").write_text(
        "# comment\n\nAcademind\nApp Router\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_transcribe(
        input_path: Path, output_dir: Path, *, overwrite: bool, context: str = ""
    ):
        assert context == "Academind App Router"
        paths = [
            output_dir / "talk.txt",
            output_dir / "talk.words.json",
            output_dir / "talk.srt",
            output_dir / "talk.vtt",
        ]
        for p in paths:
            p.write_text("ok\n", encoding="utf-8")
        return paths

    monkeypatch.setattr("stt.cli.transcribe_file", fake_transcribe)
    code = run([str(media), "-o", str(out)])
    assert code == 0
    err = capsys.readouterr().err
    assert "loaded 2 terms from terms.txt" in err


def test_explicit_terms_path_missing_fails(tmp_path: Path):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    missing = tmp_path / "missing-terms.txt"
    code = run([str(media), "-o", str(tmp_path / "out"), "--terms", str(missing)])
    assert code == 1


def test_explicit_terms_path_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"
    terms = tmp_path / "custom.txt"
    terms.write_text("Next.js\nMaximilian Schwarzmüller\n", encoding="utf-8")

    def fake_transcribe(
        input_path: Path, output_dir: Path, *, overwrite: bool, context: str = ""
    ):
        assert context == "Next.js Maximilian Schwarzmüller"
        paths = [
            output_dir / "talk.txt",
            output_dir / "talk.words.json",
            output_dir / "talk.srt",
            output_dir / "talk.vtt",
        ]
        for p in paths:
            p.write_text("ok\n", encoding="utf-8")
        return paths

    monkeypatch.setattr("stt.cli.transcribe_file", fake_transcribe)
    code = run([str(media), "-o", str(out), "--terms", str(terms)])
    assert code == 0
