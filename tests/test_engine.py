"""Model-free engine tests: snapshot resolution, one-call inference, validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from stt.engine import (
    ALIGNER_MODEL_ID,
    ALIGNER_REVISION,
    ASR_MODEL_ID,
    ASR_REVISION,
    AlignmentUnit,
    EngineError,
    ResultValidationError,
    SnapshotCacheError,
    Transcription,
    normalize_result,
    resolve_snapshot,
    run_transcription,
    transcribe_file,
)
from stt.outputs import OutputCollisionError


@dataclass(frozen=True)
class FakeResult:
    text: str
    language: str
    segments: list[dict[str, Any]] | None = None
    truncated: bool = False


# ---------------------------------------------------------------------------
# resolve_snapshot
# ---------------------------------------------------------------------------


def test_resolve_snapshot_cache_hit_no_online(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    cached = tmp_path / "cached-model"
    cached.mkdir()
    calls: list[dict[str, Any]] = []

    def fake_download(*, repo_id: str, revision: str, local_files_only: bool):
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_files_only": local_files_only,
            }
        )
        assert local_files_only is True
        return str(cached)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    path = resolve_snapshot(ASR_MODEL_ID, ASR_REVISION)
    assert path == cached
    assert calls == [
        {
            "repo_id": ASR_MODEL_ID,
            "revision": ASR_REVISION,
            "local_files_only": True,
        }
    ]


def test_resolve_snapshot_online_retry_only_on_cache_miss(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from huggingface_hub.errors import LocalEntryNotFoundError

    online = tmp_path / "online-model"
    online.mkdir()
    calls: list[bool] = []

    def fake_download(*, repo_id: str, revision: str, local_files_only: bool):
        calls.append(local_files_only)
        assert repo_id == ALIGNER_MODEL_ID
        assert revision == ALIGNER_REVISION
        if local_files_only:
            raise LocalEntryNotFoundError("not in cache")
        return str(online)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    path = resolve_snapshot(ALIGNER_MODEL_ID, ALIGNER_REVISION)
    assert path == online
    assert calls == [True, False]


def test_resolve_snapshot_online_retry_on_incomplete_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from huggingface_hub.errors import IncompleteSnapshotError

    online = tmp_path / "online-model"
    online.mkdir()
    calls: list[bool] = []

    def fake_download(*, repo_id: str, revision: str, local_files_only: bool):
        calls.append(local_files_only)
        if local_files_only:
            raise IncompleteSnapshotError("incomplete", str(tmp_path / "partial"))
        return str(online)

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert resolve_snapshot(ASR_MODEL_ID, ASR_REVISION) == online
    assert calls == [True, False]


def test_resolve_snapshot_warm_success_when_online_forced_to_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Cache hit must not attempt online download even if online would fail."""
    cached = tmp_path / "warm"
    cached.mkdir()

    def fake_download(*, repo_id: str, revision: str, local_files_only: bool):
        if local_files_only:
            return str(cached)
        raise RuntimeError("network must not be used on warm cache hit")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    assert resolve_snapshot(ASR_MODEL_ID, ASR_REVISION) == cached


def test_resolve_snapshot_offline_env_blocks_online_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls: list[bool] = []

    def fake_download(*, repo_id: str, revision: str, local_files_only: bool):
        calls.append(local_files_only)
        if local_files_only:
            raise LocalEntryNotFoundError("cache miss")
        raise AssertionError("online retry must not run under HF_HUB_OFFLINE=1")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    with pytest.raises(SnapshotCacheError, match="HF_HUB_OFFLINE=1"):
        resolve_snapshot(ASR_MODEL_ID, ASR_REVISION)
    assert calls == [True]


def test_resolve_snapshot_exact_repo_revision_constants():
    assert ASR_MODEL_ID == "Qwen/Qwen3-ASR-1.7B"
    assert ASR_REVISION == "7278e1e70fe206f11671096ffdd38061171dd6e5"
    assert ALIGNER_MODEL_ID == "Qwen/Qwen3-ForcedAligner-0.6B"
    assert ALIGNER_REVISION == "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"


# ---------------------------------------------------------------------------
# normalize_result — accept
# ---------------------------------------------------------------------------


def test_normalize_empty_text_none_segments():
    result = normalize_result(FakeResult(text="", language="English", segments=None))
    assert result == Transcription(text="", language="English", units=())


def test_normalize_empty_text_empty_list_segments():
    result = normalize_result(FakeResult(text="", language="English", segments=[]))
    assert result == Transcription(text="", language="English", units=())


def test_normalize_valid_segments_millisecond_conversion():
    raw = FakeResult(
        text="Hello world",
        language="English",
        segments=[
            {"text": "Hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.4, "end": 0.9},
        ],
    )
    result = normalize_result(raw)
    assert result.text == "Hello world"
    assert result.language == "English"
    assert result.units == (
        AlignmentUnit(text="Hello", start_ms=0, end_ms=400),
        AlignmentUnit(text="world", start_ms=400, end_ms=900),
    )


def test_normalize_equal_boundaries_ok():
    raw = FakeResult(
        text="ab",
        language="English",
        segments=[
            {"text": "a", "start": 1.0, "end": 1.0},
            {"text": "b", "start": 1.0, "end": 1.5},
        ],
    )
    result = normalize_result(raw)
    assert result.units[0].end_ms == result.units[1].start_ms == 1000


def test_normalize_rounding_to_ms():
    raw = FakeResult(
        text="x",
        language="English",
        segments=[{"text": "x", "start": 0.0014, "end": 0.0016}],
    )
    result = normalize_result(raw)
    assert result.units[0].start_ms == 1
    assert result.units[0].end_ms == 2


# ---------------------------------------------------------------------------
# normalize_result — reject
# ---------------------------------------------------------------------------


def test_reject_truncated_true():
    with pytest.raises(ResultValidationError, match="truncated"):
        normalize_result(
            FakeResult(
                text="hi",
                language="English",
                segments=[{"text": "hi", "start": 0.0, "end": 0.1}],
                truncated=True,
            )
        )


def test_reject_language_mismatch():
    with pytest.raises(ResultValidationError, match="language"):
        normalize_result(
            FakeResult(
                text="hola",
                language="Spanish",
                segments=[{"text": "hola", "start": 0.0, "end": 0.1}],
            )
        )


def test_reject_unknown_language():
    with pytest.raises(ResultValidationError, match="language"):
        normalize_result(
            FakeResult(
                text="x",
                language="unknown",
                segments=[{"text": "x", "start": 0.0, "end": 0.1}],
            )
        )


def test_reject_boolean_timestamps():
    with pytest.raises(ResultValidationError, match="start"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": True, "end": 0.1}],
            )
        )
    with pytest.raises(ResultValidationError, match="end"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": 0.0, "end": False}],
            )
        )


def test_reject_string_timestamps():
    with pytest.raises(ResultValidationError, match="start"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": "0.0", "end": 0.1}],
            )
        )


def test_reject_missing_segment_keys():
    with pytest.raises(ResultValidationError, match="missing keys"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": 0.0}],
            )
        )


def test_reject_nan_inf_timestamps():
    with pytest.raises(ResultValidationError, match="start"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": float("nan"), "end": 0.1}],
            )
        )
    with pytest.raises(ResultValidationError, match="end"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": 0.0, "end": float("inf")}],
            )
        )


def test_reject_negative_times():
    with pytest.raises(ResultValidationError, match="negative"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": -0.1, "end": 0.1}],
            )
        )


def test_reject_reversed_times():
    with pytest.raises(ResultValidationError, match="reversed"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": 0.5, "end": 0.1}],
            )
        )


def test_reject_overlapping_times():
    with pytest.raises(ResultValidationError, match="overlaps"):
        normalize_result(
            FakeResult(
                text="a b",
                language="English",
                segments=[
                    {"text": "a", "start": 0.0, "end": 0.5},
                    {"text": "b", "start": 0.4, "end": 0.8},
                ],
            )
        )


def test_reject_post_rounding_invalidity():
    """Values that round to a reversed or negative millisecond pair are rejected."""
    # 0.0004 → 0 ms, -0.0004 → 0 ms is ok equal; use values that reverse after round
    # start=0.0006→1, end=0.0004→0 → reversed after conversion
    with pytest.raises(ResultValidationError, match="reversed"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": "x", "start": 0.0006, "end": 0.0004}],
            )
        )


def test_reject_empty_text_with_segments():
    with pytest.raises(ResultValidationError, match="empty text"):
        normalize_result(
            FakeResult(
                text="",
                language="English",
                segments=[{"text": "x", "start": 0.0, "end": 0.1}],
            )
        )


def test_reject_whitespace_only_text():
    with pytest.raises(ResultValidationError, match="whitespace-only"):
        normalize_result(
            FakeResult(text="   \n\t  ", language="English", segments=None)
        )


def test_reject_punctuation_only_no_alignable_units():
    with pytest.raises(ResultValidationError, match="punctuation-only"):
        normalize_result(
            FakeResult(text="... !!! —", language="English", segments=None)
        )
    with pytest.raises(ResultValidationError, match="punctuation-only"):
        normalize_result(
            FakeResult(
                text="?!?",
                language="English",
                segments=[{"text": "?", "start": 0.0, "end": 0.1}],
            )
        )


def test_reject_nonempty_text_without_segments():
    with pytest.raises(ResultValidationError, match="complete segment"):
        normalize_result(FakeResult(text="Hello", language="English", segments=None))
    with pytest.raises(ResultValidationError, match="complete segment"):
        normalize_result(FakeResult(text="Hello", language="English", segments=[]))


def test_reject_non_mapping_segment():
    with pytest.raises(ResultValidationError, match="mapping"):
        normalize_result(
            FakeResult(text="x", language="English", segments=["bad"])  # type: ignore[list-item]
        )


def test_reject_non_str_segment_text():
    with pytest.raises(ResultValidationError, match="text must be str"):
        normalize_result(
            FakeResult(
                text="x",
                language="English",
                segments=[{"text": 1, "start": 0.0, "end": 0.1}],
            )
        )


# ---------------------------------------------------------------------------
# run_transcription / one-call contract
# ---------------------------------------------------------------------------


def test_run_transcription_calls_mlx_once_with_exact_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    asr = tmp_path / "asr"
    aligner = tmp_path / "aligner"
    asr.mkdir()
    aligner.mkdir()

    def fake_resolve(repo_id: str, revision: str) -> Path:
        if repo_id == ASR_MODEL_ID:
            assert revision == ASR_REVISION
            return asr
        if repo_id == ALIGNER_MODEL_ID:
            assert revision == ALIGNER_REVISION
            return aligner
        raise AssertionError(f"unexpected repo {repo_id}")

    mlx_transcribe = MagicMock(
        return_value=FakeResult(
            text="Hi",
            language="English",
            segments=[{"text": "Hi", "start": 0.0, "end": 0.2}],
        )
    )

    monkeypatch.setattr("stt.engine.resolve_snapshot", fake_resolve)
    monkeypatch.setattr("mlx_qwen3_asr.transcribe", mlx_transcribe)

    # Patch the import path used inside _call_transcribe
    import mlx_qwen3_asr

    monkeypatch.setattr(mlx_qwen3_asr, "transcribe", mlx_transcribe)

    result = run_transcription(media)
    assert result.text == "Hi"
    assert result.units == (AlignmentUnit(text="Hi", start_ms=0, end_ms=200),)
    mlx_transcribe.assert_called_once_with(
        media,
        model=str(asr),
        language="English",
        return_timestamps=True,
        forced_aligner=str(aligner),
        context="",
    )


def test_run_transcription_passes_context_to_mlx(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    asr = tmp_path / "asr"
    aligner = tmp_path / "aligner"
    asr.mkdir()
    aligner.mkdir()

    def fake_resolve(repo_id: str, revision: str) -> Path:
        return asr if repo_id == ASR_MODEL_ID else aligner

    mlx_transcribe = MagicMock(
        return_value=FakeResult(
            text="Hi",
            language="English",
            segments=[{"text": "Hi", "start": 0.0, "end": 0.2}],
        )
    )

    monkeypatch.setattr("stt.engine.resolve_snapshot", fake_resolve)
    import mlx_qwen3_asr

    monkeypatch.setattr(mlx_qwen3_asr, "transcribe", mlx_transcribe)

    run_transcription(media, context="Academind App Router")
    mlx_transcribe.assert_called_once_with(
        media,
        model=str(asr),
        language="English",
        return_timestamps=True,
        forced_aligner=str(aligner),
        context="Academind App Router",
    )


def test_run_transcription_preserves_upstream_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")

    monkeypatch.setattr(
        "stt.engine.resolve_snapshot",
        lambda repo_id, revision: tmp_path / repo_id.replace("/", "_"),
    )

    def boom(*_a, **_k):
        raise RuntimeError("mlx exploded")

    import mlx_qwen3_asr

    monkeypatch.setattr(mlx_qwen3_asr, "transcribe", boom)

    with pytest.raises(EngineError, match="transcription failed"):
        run_transcription(media)


# ---------------------------------------------------------------------------
# transcribe_file orchestration: preflight before resolve
# ---------------------------------------------------------------------------


def test_transcribe_file_preflight_before_model_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    # Create a colliding target
    (out / "talk.txt").write_text("existing\n", encoding="utf-8")

    resolve_calls: list[tuple[str, str]] = []

    def tracking_resolve(repo_id: str, revision: str) -> Path:
        resolve_calls.append((repo_id, revision))
        return tmp_path / "model"

    monkeypatch.setattr("stt.engine.resolve_snapshot", tracking_resolve)

    with pytest.raises(OutputCollisionError, match="already exists"):
        transcribe_file(media, out, overwrite=False)

    assert resolve_calls == [], (
        "model resolution must not run after collision preflight"
    )


def test_transcribe_file_preflight_passes_then_runs_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    media = tmp_path / "talk.wav"
    media.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()

    validated = Transcription(
        text="Hi",
        language="English",
        units=(AlignmentUnit(text="Hi", start_ms=0, end_ms=100),),
    )
    published = [
        out / "talk.txt",
        out / "talk.words.json",
        out / "talk.srt",
        out / "talk.vtt",
    ]

    seen: dict[str, str] = {}

    def fake_run(path: Path, *, context: str = "") -> Transcription:
        seen["context"] = context
        return validated

    monkeypatch.setattr("stt.engine.run_transcription", fake_run)
    monkeypatch.setattr(
        "stt.outputs.publish_artifacts",
        lambda input_path, output_dir, result, *, overwrite: published,
    )

    paths = transcribe_file(media, out, overwrite=False, context="foo bar")
    assert paths == published
    assert seen["context"] == "foo bar"
    err = capsys.readouterr().err
    assert "stt: checking output paths…" in err
    assert "stt: writing artifacts (1 timed unit)…" in err
    assert "stt: done" in err


def test_run_transcription_emits_phase_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"fake")
    asr = tmp_path / "asr"
    aligner = tmp_path / "aligner"
    asr.mkdir()
    aligner.mkdir()

    def fake_resolve(repo_id: str, revision: str) -> Path:
        return asr if repo_id == ASR_MODEL_ID else aligner

    import mlx_qwen3_asr

    monkeypatch.setattr("stt.engine.resolve_snapshot", fake_resolve)
    monkeypatch.setattr(
        mlx_qwen3_asr,
        "transcribe",
        lambda *a, **k: FakeResult(
            text="Hi",
            language="English",
            segments=[{"text": "Hi", "start": 0.0, "end": 0.2}],
        ),
    )

    run_transcription(media)
    err = capsys.readouterr().err
    assert "stt: resolving ASR model (cache-first)…" in err
    assert "stt: resolving forced-aligner model (cache-first)…" in err
    assert "stt: transcribing + aligning" in err
    assert "stt: validating transcription result…" in err
