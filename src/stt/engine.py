"""Model resolution, one-shot inference, and result validation."""

from __future__ import annotations

import math
import os
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Immutable pinned model identities (smoke-verified revisions).
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"

_FORCED_LANGUAGE = "English"


class EngineError(Exception):
    """Base error for engine failures surfaced as concise CLI errors."""


class SnapshotCacheError(EngineError):
    """Model snapshot is not available in the local cache (and online retry is blocked)."""


class ResultValidationError(EngineError):
    """Upstream transcription result failed local validation."""


@dataclass(frozen=True)
class AlignmentUnit:
    """One forced-aligner unit with millisecond timestamps."""

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class Transcription:
    """Validated local transcription record.

    Notes:
        Silent media is not guaranteed to yield an empty result; hallucinated
        text remains model output and is not rewritten here.
    """

    text: str
    language: str
    units: tuple[AlignmentUnit, ...]


def resolve_snapshot(repo_id: str, revision: str) -> Path:
    """Resolve a pinned snapshot cache-first, retrying online only on cache miss.

    Tries ``local_files_only=True`` first. On cache-miss / incomplete-snapshot
    errors only, retries the same repo/revision online unless ``HF_HUB_OFFLINE=1``.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        IncompleteSnapshotError,
        LocalEntryNotFoundError,
    )

    try:
        path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
        )
        return Path(path)
    except (LocalEntryNotFoundError, IncompleteSnapshotError) as cache_exc:
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            raise SnapshotCacheError(
                f"model cache miss for {repo_id}@{revision} (HF_HUB_OFFLINE=1)"
            ) from cache_exc
        try:
            path = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_files_only=False,
            )
            return Path(path)
        except Exception as online_exc:
            raise EngineError(
                f"failed to download model {repo_id}@{revision}: {online_exc}"
            ) from online_exc


def _is_kept_aligner_char(ch: str) -> bool:
    """Mirror upstream English aligner cleaner keep-set (L*/N* + ASCII apostrophe)."""
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N"))


def _has_alignable_unit(text: str) -> bool:
    """True if any whitespace-split token retains aligner-kept characters."""
    for token in text.split():
        cleaned = "".join(ch for ch in token if _is_kept_aligner_char(ch))
        if cleaned:
            return True
    return False


def _is_real_number(value: Any) -> bool:
    """Accept finite int/float seconds; reject bool and non-numeric types."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _seconds_to_ms(value: Any) -> int:
    # Plan contract: int(round(seconds * 1000)); round() already returns int.
    return round(float(value) * 1000)


def _require_mapping(segment: Any, index: int) -> Mapping[str, Any]:
    if not isinstance(segment, Mapping):
        raise ResultValidationError(
            f"segment[{index}] must be a mapping, got {type(segment).__name__}"
        )
    return segment


def normalize_result(raw: Any) -> Transcription:
    """Validate an upstream transcription result and convert to frozen local records."""
    if raw is None:
        raise ResultValidationError("transcription result is missing")

    try:
        text = raw.text
        language = raw.language
        segments = raw.segments
        truncated = getattr(raw, "truncated", False)
    except AttributeError as exc:
        raise ResultValidationError(
            f"transcription result missing required field: {exc}"
        ) from exc

    if truncated is True:
        raise ResultValidationError("transcription result is truncated")

    if not isinstance(text, str):
        raise ResultValidationError(
            f"transcription text must be str, got {type(text).__name__}"
        )
    if not isinstance(language, str):
        raise ResultValidationError(
            f"transcription language must be str, got {type(language).__name__}"
        )
    if language != _FORCED_LANGUAGE:
        raise ResultValidationError(
            f"expected language {_FORCED_LANGUAGE!r}, got {language!r}"
        )

    # Exact empty speech: text == "" with segments None or [].
    if text == "":
        if segments is None or segments == []:
            return Transcription(text="", language=_FORCED_LANGUAGE, units=())
        raise ResultValidationError("empty text must not include segments")

    if text.strip() == "":
        raise ResultValidationError("whitespace-only transcript text is rejected")

    if not _has_alignable_unit(text):
        raise ResultValidationError(
            "punctuation-only transcript text has no alignable units"
        )

    if segments is None:
        raise ResultValidationError(
            "nonempty text requires a complete segment sequence"
        )
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise ResultValidationError(
            f"segments must be a sequence, got {type(segments).__name__}"
        )
    if len(segments) == 0:
        raise ResultValidationError(
            "nonempty text requires a complete segment sequence"
        )

    units: list[AlignmentUnit] = []
    for index, segment in enumerate(segments):
        seg = _require_mapping(segment, index)
        missing = [key for key in ("text", "start", "end") if key not in seg]
        if missing:
            raise ResultValidationError(
                f"segment[{index}] missing keys: {', '.join(missing)}"
            )
        seg_text = seg["text"]
        if not isinstance(seg_text, str):
            raise ResultValidationError(
                f"segment[{index}].text must be str, got {type(seg_text).__name__}"
            )
        start = seg["start"]
        end = seg["end"]
        if not _is_real_number(start):
            raise ResultValidationError(
                f"segment[{index}].start must be a finite number, got {start!r}"
            )
        if not _is_real_number(end):
            raise ResultValidationError(
                f"segment[{index}].end must be a finite number, got {end!r}"
            )
        start_ms = _seconds_to_ms(start)
        end_ms = _seconds_to_ms(end)
        if start_ms < 0 or end_ms < 0:
            raise ResultValidationError(
                f"segment[{index}] has negative timestamp after conversion "
                f"({start_ms}, {end_ms})"
            )
        if start_ms > end_ms:
            raise ResultValidationError(
                f"segment[{index}] has reversed times after conversion "
                f"({start_ms} > {end_ms})"
            )
        if units and units[-1].end_ms > start_ms:
            raise ResultValidationError(
                f"segment[{index}] overlaps previous unit "
                f"({units[-1].end_ms} > {start_ms})"
            )
        units.append(AlignmentUnit(text=seg_text, start_ms=start_ms, end_ms=end_ms))

    return Transcription(
        text=text,
        language=_FORCED_LANGUAGE,
        units=tuple(units),
    )


def _call_transcribe(
    input_path: Path,
    asr_path: Path,
    aligner_path: Path,
    *,
    context: str = "",
) -> Any:
    """Invoke mlx_qwen3_asr.transcribe exactly once with fixed English/timestamp args."""
    from mlx_qwen3_asr import transcribe as mlx_transcribe

    return mlx_transcribe(
        input_path,
        model=str(asr_path),
        language=_FORCED_LANGUAGE,
        return_timestamps=True,
        forced_aligner=str(aligner_path),
        context=context,
    )


def _progress(message: str) -> None:
    """Emit a phase update to stderr immediately (stdout stays artifact paths only)."""
    import sys

    print(f"stt: {message}", file=sys.stderr, flush=True)


def run_transcription(input_path: Path, *, context: str = "") -> Transcription:
    """Resolve pinned snapshots, run one inference call, and validate the result."""
    _progress("resolving ASR model (cache-first)…")
    asr_path = resolve_snapshot(ASR_MODEL_ID, ASR_REVISION)
    _progress(f"ASR ready: {asr_path}")

    _progress("resolving forced-aligner model (cache-first)…")
    aligner_path = resolve_snapshot(ALIGNER_MODEL_ID, ALIGNER_REVISION)
    _progress(f"aligner ready: {aligner_path}")

    if context:
        _progress(f"using domain context ({len(context.split())} whitespace tokens)…")
    else:
        _progress("no domain context terms")

    _progress(
        "transcribing + aligning (often the long step; loads models, then runs MLX)…"
    )
    try:
        raw = _call_transcribe(
            input_path,
            asr_path,
            aligner_path,
            context=context,
        )
    except EngineError:
        raise
    except Exception as exc:
        raise EngineError(f"transcription failed: {exc}") from exc

    _progress("validating transcription result…")
    return normalize_result(raw)


def transcribe_file(
    input_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    context: str = "",
) -> list[Path]:
    """Orchestrate preflight → resolve → one inference → validate → publish.

    Collision preflight runs before any model resolution. Caption restoration
    and atomic multi-artifact publish run after a validated Transcription.

    ``context`` is optional domain vocabulary passed to Qwen3-ASR (space-joined
    terms from a terms file).
    """
    from stt.outputs import preflight_targets, publish_artifacts, target_paths

    _progress("checking output paths…")
    targets = target_paths(input_path, output_dir)
    preflight_targets(input_path, targets, overwrite=overwrite)

    result = run_transcription(input_path, context=context)

    unit_count = len(result.units)
    _progress(
        f"writing artifacts ({unit_count} timed unit{'s' if unit_count != 1 else ''})…"
    )
    paths = publish_artifacts(
        input_path,
        output_dir,
        result,
        overwrite=overwrite,
    )
    _progress("done")
    return paths
