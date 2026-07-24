"""argparse CLI for English Qwen3 speech-to-text."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stt",
        description=(
            "Transcribe one English audio/video file with Qwen3-ASR and emit "
            "TXT, words JSON, SRT, and VTT artifacts."
        ),
    )
    parser.add_argument(
        "INPUT",
        type=Path,
        help="Path to a single audio or video file ffmpeg can decode.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output artifacts (default: current directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output artifacts with the same names.",
    )
    return parser


def _apple_silicon_hint(exc: BaseException) -> str:
    return (
        f"{exc}\n"
        "This CLI requires Apple Silicon (arm64 macOS) with a working MLX runtime."
    )


def transcribe_file(
    input_path: Path, output_dir: Path, *, overwrite: bool
) -> list[Path]:
    """Run inference and publish artifacts.

    Imported/called only after argument validation. Tests monkeypatch this
    boundary so model loading is never required for CLI unit tests.
    """
    # Lazy import keeps MLX/runtime off the help and validation paths.
    try:
        from stt.engine import transcribe_file as _engine_transcribe
    except ImportError as exc:
        raise RuntimeError(_apple_silicon_hint(exc)) from exc

    try:
        return _engine_transcribe(input_path, output_dir, overwrite=overwrite)
    except Exception as exc:
        # Preserve root cause; attach platform hint only for import/runtime MLX failures.
        name = type(exc).__module__ + "." + type(exc).__name__
        if "mlx" in name.lower() or "mlx" in str(exc).lower():
            raise RuntimeError(_apple_silicon_hint(exc)) from exc
        raise


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.INPUT
    output_dir: Path = args.output_dir
    overwrite: bool = args.overwrite

    if not input_path.is_file():
        if input_path.exists() and input_path.is_dir():
            print(
                f"error: INPUT is a directory, expected a file: {input_path}",
                file=sys.stderr,
            )
        else:
            print(f"error: INPUT is not a file: {input_path}", file=sys.stderr)
        return 1

    if output_dir.exists() and not output_dir.is_dir():
        print(
            f"error: output path exists and is not a directory: {output_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        paths = transcribe_file(input_path, output_dir, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 — CLI boundary: any failure is a concise nonzero exit
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for path in paths:
        print(path)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
