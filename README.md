# stt — English Qwen3 speech-to-text CLI

Local Apple Silicon CLI that runs **Qwen3-ASR-1.7B** once per English media file and writes four punctuated artifacts:

| Artifact | Description |
|---|---|
| `<stem>.txt` | Canonical transcript + one trailing newline |
| `<stem>.words.json` | Schema-v1 timed alignment units with restored punctuation |
| `<stem>.srt` | SubRip captions |
| `<stem>.vtt` | WebVTT captions |

## Requirements

- Apple Silicon macOS (MLX)
- Python 3.12 (managed by uv)
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` on `PATH` (any format ffmpeg can decode)
- ~**6.1 GiB** disk for the pinned ASR + forced-aligner Hugging Face snapshots (downloaded on first use into the normal HF cache)

## Install

```bash
uv sync --frozen --dev
```

## Usage

```bash
uv run stt INPUT [-o DIR] [--overwrite]
```

- `INPUT` — one audio/video file
- `-o` / `--output-dir` — output directory (default: `.`); created if missing
- `--overwrite` — replace existing artifact files with the same names

English is always forced. All four formats are always written. There are no model, language, format, streaming, diarization, or batch flags.

Example:

```bash
uv run stt interview.wav -o ./transcripts
```

On success, the four absolute-or-relative artifact paths are printed to **stdout** (one per line). Progress and errors go to **stderr** (flushed phase lines such as resolving models, transcribing/aligning, writing artifacts). Exit code is `0` only after all four files are committed.

Typical stderr phases:

```text
stt: checking output paths…
stt: resolving ASR model (cache-first)…
stt: ASR ready: …
stt: resolving forced-aligner model (cache-first)…
stt: aligner ready: …
stt: transcribing + aligning (often the long step; loads models, then runs MLX)…
stt: validating transcription result…
stt: writing artifacts (N timed units)…
stt: done
```

## Models and cache

Pinned immutable revisions (first resolved cache-first, then online only on cache miss):

- `Qwen/Qwen3-ASR-1.7B` @ `7278e1e70fe206f11671096ffdd38061171dd6e5`
- `Qwen/Qwen3-ForcedAligner-0.6B` @ `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`

**Cold run:** downloads into the Hugging Face hub cache (~6.1 GiB total).  
**Warm run:** uses local snapshots only; no network required.

To forbid network access after a warm cache exists:

```bash
HF_HUB_OFFLINE=1 uv run stt interview.wav -o ./out --overwrite
```

A cache miss with `HF_HUB_OFFLINE=1` fails with a concise error (no download attempt).

## Output schema and bytes

### TXT

Canonical `TranscriptionResult.text` plus exactly one final LF.  
Empty speech (accepted empty result): `b"\n"`.

### `.words.json` (schema version 1)

UTF-8, `ensure_ascii=False`, indent 2, trailing LF. Top-level keys **exactly**:

```json
{
  "schema_version": 1,
  "text": "<canonical transcript>",
  "language": "English",
  "words": [
    {"text": "<display unit>", "start_ms": 0, "end_ms": 120}
  ]
}
```

- `schema_version` is integer `1`
- `language` is always `"English"`
- `words[]` maps **1:1 to forced-aligner units** (not guaranteed linguistic words)
- each `words[].text` includes that unit’s owned source punctuation/capitalization
- root `text` is the only lossless canonical transcript
- timestamps are integer milliseconds, monotonic non-overlapping (`start_ms <= end_ms <= next.start_ms`)

Empty result:

```json
{
  "schema_version": 1,
  "text": "",
  "language": "English",
  "words": []
}
```

### SRT / VTT

Built from the same restored units and fixed cue-grouping policy (max 10 units, 42 display chars, 6000 ms, break on gaps ≥ 800 ms and sentence-ending punctuation).  
Empty result: SRT `b""`, VTT `b"WEBVTT\n\n"`.

## Overwrite and collision semantics

- Before any model work, if any target path already exists and `--overwrite` is not set, the CLI fails.
- A target that is a directory is always rejected.
- A target that aliases the input path (resolved path equality or `samefile`) is **always** rejected, even with `--overwrite`.
- Payloads are fully rendered in memory, written to same-directory temp files (`.*.tmp-*`), then committed:
  - with `--overwrite`: `os.replace(temp, target)`
  - without: `os.link(temp, target)` then unlink temp (atomic no-clobber; a racer created after preflight is never overwritten)
- No target is ever visible with partial bytes. A mid-commit failure may leave a **subset** of complete artifacts (no cross-file transaction). Crash durability / `fsync` are not claimed. Temps are cleaned on failure.

## English-only punctuation guarantee

Qwen’s punctuated transcript is canonical. Exact English reconciliation maps that source text onto forced-alignment units (mirroring the upstream aligner cleaner) before cue grouping, so SRT/VTT/JSON unit text keep punctuation and capitalization. Mismatch is a hard failure — no fuzzy recovery and no punctuation-stripped fallback.

This ownership policy is **English-only**. Multilingual / CJK handling is intentionally deferred.

## Deferred features

Not in scope: multilingual tokenization, batch inputs, streaming/microphone capture, diarization, server/API mode, model/backend selection, quantization, transcript grammar rewriting, one-word-per-cue subtitles, configurable caption policies, doctor/download subcommands, air-gapped model packaging, cross-file transactional commits, crash-durability guarantees.

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

### Real-model Mac smoke (optional)

Requires network once for the official sample WAV and (if cold) model download:

```bash
# see plan T4 for the full checksummed sequence; validator:
uv run python scripts/validate_smoke.py /path/to/output-dir
```

The smoke validator compares against a pinned independent oracle transcript and prints `SMOKE OK`.
