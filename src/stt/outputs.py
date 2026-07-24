"""Output path helpers, artifact rendering, and atomic publishing."""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from stt.captions import group_cues, render_srt, render_vtt, restore_source_units

if TYPE_CHECKING:
    from stt.captions import SourceUnit
    from stt.engine import Transcription

ARTIFACT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("txt", ".txt"),
    ("words_json", ".words.json"),
    ("srt", ".srt"),
    ("vtt", ".vtt"),
)

_ORDERED_KEYS: tuple[str, ...] = tuple(key for key, _ in ARTIFACT_SUFFIXES)


class OutputCollisionError(Exception):
    """Raised when an output target cannot be written safely."""


class PublishError(Exception):
    """Raised when artifact rendering or commit fails."""


def target_paths(input_path: Path, output_dir: Path) -> dict[str, Path]:
    """Derive the four artifact paths from the input stem and output directory."""
    stem = input_path.stem
    return {key: output_dir / f"{stem}{suffix}" for key, suffix in ARTIFACT_SUFFIXES}


def _paths_alias(input_path: Path, target: Path) -> bool:
    """True when target is the same path or same filesystem object as input."""
    try:
        if input_path.resolve() == target.resolve():
            return True
    except OSError:
        pass
    try:
        if input_path.exists() and target.exists() and input_path.samefile(target):
            return True
    except OSError:
        return False
    return False


def preflight_targets(
    input_path: Path,
    targets: Mapping[str, Path],
    *,
    overwrite: bool,
) -> None:
    """Reject unsafe or colliding targets before model resolution.

    - Any target that aliases the input is always rejected (even with overwrite).
    - A target path that exists as a directory is rejected.
    - An existing file target is rejected unless overwrite is True.
    """
    for target in targets.values():
        if _paths_alias(input_path, target):
            raise OutputCollisionError(f"output target aliases input path: {target}")
        if target.exists():
            if target.is_dir():
                raise OutputCollisionError(
                    f"output target exists and is a directory: {target}"
                )
            if not overwrite:
                raise OutputCollisionError(
                    f"output target already exists (use --overwrite): {target}"
                )


def render_txt(result: Transcription) -> bytes:
    """Canonical transcript text plus exactly one trailing LF."""
    return f"{result.text}\n".encode()


def render_words_json(
    result: Transcription,
    units: tuple[SourceUnit, ...] | list[SourceUnit],
) -> bytes:
    """Schema-v1 words JSON: UTF-8, ensure_ascii=False, indent 2, trailing LF."""
    payload = {
        "schema_version": 1,
        "text": result.text,
        "language": "English",
        "words": [
            {
                "text": unit.text,
                "start_ms": unit.start_ms,
                "end_ms": unit.end_ms,
            }
            for unit in units
        ],
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()


def render_payloads(result: Transcription) -> dict[str, bytes]:
    """Restore source units, group cues, and render all four artifact payloads."""
    source_units = restore_source_units(result)
    cues = group_cues(source_units, result.text)
    return {
        "txt": render_txt(result),
        "words_json": render_words_json(result, source_units),
        "srt": render_srt(cues).encode(),
        "vtt": render_vtt(cues).encode(),
    }


def _temp_for(target: Path) -> Path:
    """Same-directory temp path matching ``.*.tmp-*`` cleanup glob."""
    token = secrets.token_hex(8)
    return target.parent / f".{target.name}.tmp-{token}"


def _write_bytes(path: Path, data: bytes) -> None:
    """Write complete payload bytes to a temporary path."""
    path.write_bytes(data)


def _commit_one(temp: Path, target: Path, *, overwrite: bool) -> None:
    """Atomically publish one complete temp file to its target path."""
    if target.exists() and target.is_dir():
        raise OutputCollisionError(f"output target exists and is a directory: {target}")
    if overwrite:
        try:
            os.replace(temp, target)
        except OSError as exc:
            raise PublishError(f"failed to commit artifact {target}: {exc}") from exc
        return
    try:
        os.link(temp, target)
    except FileExistsError as exc:
        raise OutputCollisionError(
            f"output target already exists (use --overwrite): {target}"
        ) from exc
    except OSError as exc:
        raise PublishError(f"failed to commit artifact {target}: {exc}") from exc
    try:
        temp.unlink()
    except OSError as exc:
        raise PublishError(
            f"committed {target} but failed to remove temp {temp}: {exc}"
        ) from exc


def _cleanup_temps(temps: Mapping[str, Path]) -> None:
    for temp in temps.values():
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def publish_artifacts(
    input_path: Path,
    output_dir: Path,
    result: Transcription,
    *,
    overwrite: bool,
) -> list[Path]:
    """Render all artifacts in memory, then atomically publish each file.

    Guarantee: no target is ever visible with partial bytes. A mid-commit
    failure may leave a subset of complete artifacts (no cross-file transaction).
    Temporary files are cleaned on every failure path. Crash durability / fsync
    are intentionally out of scope.
    """
    targets = target_paths(input_path, output_dir)
    preflight_targets(input_path, targets, overwrite=overwrite)

    try:
        payloads = render_payloads(result)
    except Exception as exc:
        raise PublishError(f"failed to render artifacts: {exc}") from exc

    temps: dict[str, Path] = {}
    try:
        for key in _ORDERED_KEYS:
            target = targets[key]
            if _paths_alias(input_path, target):
                raise OutputCollisionError(
                    f"output target aliases input path: {target}"
                )
            temp = _temp_for(target)
            temps[key] = temp
            try:
                _write_bytes(temp, payloads[key])
            except OSError as exc:
                raise PublishError(
                    f"failed to write temporary artifact for {target}: {exc}"
                ) from exc

        published: list[Path] = []
        for key in _ORDERED_KEYS:
            target = targets[key]
            temp = temps[key]
            if _paths_alias(input_path, target):
                raise OutputCollisionError(
                    f"output target aliases input path: {target}"
                )
            try:
                _commit_one(temp, target, overwrite=overwrite)
            except (OutputCollisionError, PublishError):
                raise
            except OSError as exc:
                raise PublishError(
                    f"failed to commit artifact {target}: {exc}"
                ) from exc
            # Successful commit consumes the temp (replace or unlink).
            del temps[key]
            published.append(target)
        return published
    finally:
        _cleanup_temps(temps)
