# Build an English Qwen3 speech-to-text CLI

> **Status:** Ready for implementation

## Outcome and boundaries

- **Problem and target:** Build a local Apple-Silicon CLI that runs Qwen3-ASR-1.7B once per English media file and emits a punctuated transcript, punctuated timed alignment units, and punctuated SRT/VTT captions.
- **In scope:**
  - One regular input file per invocation; any audio/video format the installed ffmpeg can decode.
  - Four outputs in one chosen directory: `<stem>.txt`, `<stem>.words.json`, `<stem>.srt`, and `<stem>.vtt`.
  - Qwen’s punctuated `TranscriptionResult.text` is canonical. Exact English reconciliation maps that source text back onto forced-alignment units before cue grouping, preserving punctuation and capitalization.
  - Automatic first-use Hugging Face downloads into the normal user cache; cache-first, network-free warm execution.
  - Strict result validation, explicit overwrite behavior, atomic visibility per artifact, model-free automated tests, and one reproducible real-model Mac smoke test.
- **Out of scope:** Multilingual tokenization; batch inputs; streaming/microphone capture; diarization; server/API mode; model/backend selection; quantization; transcript grammar rewriting; one-word-per-cue subtitles; configurable caption policies; doctor/download subcommands; air-gapped model packaging; cross-file transactional commits or crash-durability guarantees.
- **Approach:** Create a small uv-managed Python 3.12 package with a stdlib `argparse` console script named `stt`. Pin `mlx-qwen3-asr==0.3.5` and immutable Qwen model revisions. Resolve models cache-first, call the Python transcription API once with English and timestamps enabled, validate and convert timestamps once, reconcile alignment units against exact transcript source spans, then render all four artifacts from one normalized result.

## Key files, evidence, and decisions

| File or source | Why it matters | Decision or plan impact |
|---|---|---|
| `pyproject.toml` | Defines package, console entry point, Python/dependency constraints, and test/lint tools. | Use Python `>=3.12,<3.13`; pin `mlx-qwen3-asr==0.3.5`; directly declare `huggingface-hub` because application code resolves immutable snapshots. |
| `uv.lock` | Captures the complete tested dependency graph. | Commit it and validate with `uv sync --frozen`. |
| `src/stt/cli.py` | Owns the small user contract and orchestration. | Support only `stt INPUT [-o DIR] [--overwrite]`; always force English and produce all four formats. |
| `src/stt/engine.py` | Isolates model/cache resolution and the upstream MLX API boundary. | Resolve immutable snapshots, call `transcribe(...)` exactly once, validate upstream fields, and expose immutable local millisecond records. |
| `src/stt/captions.py` | Owns punctuation preservation that upstream writers cannot provide. | Reproduce the English aligner cleaner exactly; use exact ordered matching and source offsets, never fuzzy matching; own cue grouping and SRT/VTT rendering. |
| `src/stt/outputs.py` | Defines durable artifact bytes and filesystem behavior. | Define schema v1, render all payloads first, protect collisions, and publish each complete artifact atomically. |
| `tests/test_engine.py` | Protects model resolution and one-call inference contract without model loads. | Mock only Hugging Face/upstream boundaries and characterize all accepted/rejected result states. |
| `tests/test_captions.py` | Protects lossy-cleaner reversal and cue behavior. | Cover punctuation, source ownership, repeated words, exact mismatch failures, and grouping boundaries. |
| `tests/test_outputs.py` | Protects schemas and filesystem safety. | Cover exact UTF-8 bytes, timestamp fields, collision races, overwrite, failed commits, and temporary cleanup. |
| `tests/test_cli.py` | Protects command behavior and exit semantics. | Exercise help, path validation, directory creation, success, and concise failures through a mocked engine boundary. |
| `scripts/validate_smoke.py` | Makes real-model acceptance repeatable without putting a 2.1 MB fixture in Git. | Validate the pinned official sample’s exact canonical transcript, schemas, timestamps, cue text, and expected artifact set; print `SMOKE OK`. |
| `README.md` | Operator setup and first-run expectations. | Document uv, ffmpeg, model cache size/download behavior, warm offline use, exact outputs, English-only guarantee, and known limits. |
| [`mlx-qwen3-asr` 0.3.5 transcription API](https://github.com/moona3k/mlx-qwen3-asr/blob/f069a0f2158b401c205c4d68633d3e3f3c5af469/mlx_qwen3_asr/transcribe.py#L224-L280) | Returns canonical text and optional timestamp segments. | Use the Python API, not the upstream CLI or stdout parsing. |
| [Aligner cleaner/tokenizer](https://github.com/moona3k/mlx-qwen3-asr/blob/f069a0f2158b401c205c4d68633d3e3f3c5af469/mlx_qwen3_asr/forced_aligner.py#L39-L91) | Keeps Unicode letters/numbers and straight apostrophes while deleting other punctuation. | Mirror this exact transformation for English source matching. A returned unit is an aligner unit, not necessarily a linguistic word. |
| [Upstream grouping policy](https://github.com/moona3k/mlx-qwen3-asr/blob/f069a0f2158b401c205c4d68633d3e3f3c5af469/mlx_qwen3_asr/writers.py#L123-L209) | Supplies useful readability defaults but normally receives punctuation-stripped units. | Apply equivalent fixed defaults only after restoring source punctuation; own formatting so SRT/VTT retain it. |
| [ASR snapshot](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5) | Immutable weights used by the verified Mac smoke test. | Pin revision `7278e1e70fe206f11671096ffdd38061171dd6e5`. |
| [Aligner snapshot](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/c7cbfc2048c462b0d63a45797104fc9db3ad62b7) | Immutable English timestamp model. | Pin revision `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`. |

## Tasks

#### T1 — Scaffold the pinned package and define the CLI contract

- **Change:**
  - Create a src-layout `pyproject.toml` and expose `stt = "stt.cli:main"`.
  - Require Python `>=3.12,<3.13`; pin `mlx-qwen3-asr==0.3.5`; directly declare `huggingface-hub>=0.20,<2`; add pytest and Ruff development dependencies; generate and commit `uv.lock`.
  - Add an `argparse` CLI accepting one `INPUT`, `--output-dir/-o` (default `.`), and opt-in `--overwrite`; import MLX/runtime code only after argument validation.
  - Require `INPUT.is_file()`. Recursively create a missing output directory before inference; reject an output path that exists but is not a directory.
  - Force English and always request all four artifacts; expose no model, language, format, streaming, diarization, or batch controls.
  - Add `.gitignore` entries for `.venv`, Python/test caches, generated output extensions, and optional project-local model caches without ignoring fixtures or expected test data.
- **Starts at:** `pyproject.toml`, `uv.lock`, `.gitignore`, `src/stt/__init__.py`, `src/stt/cli.py`, `tests/test_cli.py`
- **Verify:**
  - Run `uv sync --frozen --dev`; expect a Python 3.12 environment, no lockfile changes, and `mlx-qwen3-asr 0.3.5` installed.
  - Run `uv run stt --help`; expect exit 0 and help containing `INPUT`, `--output-dir`, and `--overwrite`, with no unsupported feature flags.
  - Run `uv run pytest -q tests/test_cli.py`; expect passing tests for help, missing input, directory-as-input rejection, nested output-directory creation, output-path-as-file rejection, and mocked success without model loading.
- **Risk/recovery:** Let unsupported MLX/platform imports retain their root cause and add one Apple-Silicon requirement hint; do not add a compatibility backend.

#### T2 — Resolve pinned models cache-first and validate one inference result

- **Change:**
  - Put both model IDs and immutable revisions in `src/stt/engine.py`.
  - Resolve each snapshot first with `snapshot_download(..., revision=..., local_files_only=True)`; on a cache-miss/incomplete-snapshot error only, retry the same repo/revision online. Allow `HF_HUB_OFFLINE=1` to prevent that retry from reaching the network and surface a concise cache-miss error.
  - Perform output collision preflight before model resolution so an invocation that cannot write does not load/download models.
  - Call `mlx_qwen3_asr.transcribe(...)` exactly once with the input, local Qwen3-ASR-1.7B path, `language="English"`, `return_timestamps=True`, and the local forced-aligner path.
  - Convert the result into frozen local records. Accept only numeric, non-boolean, finite segment `start`/`end` values; convert once with `int(round(seconds * 1000))`; then require `0 <= start_ms <= end_ms <= next.start_ms`, allowing equal boundaries.
  - Reject `truncated=True`, non-English/unknown results, malformed/missing segment keys, nonempty text without a complete segment sequence, whitespace-only text, punctuation-only text with no alignable units, empty text with segments, and every invalid timestamp sequence before rendering.
  - Accept exactly `text == ""` with `segments` equal to `None` or `[]` as an empty result. Document that silent media is not guaranteed to produce this state; hallucinated text remains model output and is not rewritten.
  - Preserve upstream download/inference/alignment failures as one concise CLI failure with a nonzero exit.
- **Starts at:** `src/stt/engine.py` (`resolve_snapshot`, `transcribe_file`, model constants, frozen result/unit records), `src/stt/cli.py`, `tests/test_engine.py`
- **Depends on:** T1
- **Verify:**
  - Run `uv run pytest -q tests/test_engine.py`; expect tests proving cache-first resolution, online retry only on cache miss, warm success when online calls are forced to fail, exact repo/revision arguments, one transcription call, and exact English/timestamp arguments.
  - In the same suite, expect accepted empty speech and rejection of booleans/strings/missing keys, NaN/Inf, negative/reversed/overlapping times, post-rounding invalidity, truncation, language mismatch, empty/nonempty state inconsistencies, and incomplete aligned prefixes.
  - Run `uv run python -c 'from importlib.metadata import version; assert version("mlx-qwen3-asr") == "0.3.5"'`; expect exit 0.
- **Risk/recovery:** Keep the adapter narrow because the dependency is pre-1.0. If its result shape changes, fail here and update characterization tests rather than spreading compatibility branches through formatters.

#### T3 — Reconcile canonical English source spans and generate cues

- **Change:**
  - Scan `result.text` into source spans without discarding any characters. For each non-whitespace source token, calculate the pinned English aligner-cleaned form by retaining Unicode `L*`/`N*` characters and ASCII `'` and deleting everything else.
  - Match every nonempty cleaned source token in exact order against every aligned unit’s text. Preserve one upstream alignment unit per match—even forms such as `state-of-the-art` or `can't—really—stop` may be one timed unit rather than multiple linguistic words. Require exact count/content equality and never use case folding, edit distance, or fuzzy recovery.
  - Treat each maximal canonical gap containing cleaned-empty punctuation tokens between two matched units as one atomic source range, so ownership can never overlap or cross. A leading gap attaches to the first unit; a trailing gap attaches to the last unit; an interstitial gap attaches wholly to the following unit when its first non-whitespace character is in the exact opening set `([{“‘«‹`, otherwise wholly to the preceding unit. This fallback covers terminal and neutral punctuation (`.!?…。！？，,;:—–-/…` and all unlisted characters) without further classification. A token such as standalone ASCII `'` is nonempty under the upstream cleaner and remains its own timed alignment unit.
  - Store each unit’s one contiguous owned canonical source range. Define its display text as `" ".join(source_slice.split())`. Build each cue’s text from the exact canonical range spanning its first through last unit, applying only that same whitespace collapse; never rebuild cue text by joining punctuation-stripped aligner text.
  - Treat whitespace outside an owned canonical range as a separator rather than an owned character. Fail reconciliation all-or-nothing on any mismatch or unowned non-whitespace source character; do not emit punctuation-stripped or partly shifted timed artifacts.
  - Group units using fixed v1 limits: maximum 10 alignment units, 42 display characters, 6,000 ms, a break at gaps `>= 800 ms`, and a sentence-ending match `[.!?…。！？](?:["'”’»)\]}]+)?$`. Cue timing spans the first unit start through the last unit end.
  - Render SRT (`HH:MM:SS,mmm`, numbered cues) and VTT (`WEBVTT`, `HH:MM:SS.mmm`) from the same cue list.
- **Starts at:** `src/stt/captions.py` (`clean_alignment_token`, `restore_source_units`, `group_cues`, `render_srt`, `render_vtt`), `tests/test_captions.py`
- **Depends on:** T2
- **Verify:**
  - Run `uv run pytest -q tests/test_captions.py`; expect exact restoration for `Hello, world!`, `I can't—really—stop.`, `It’s state-of-the-art.`, `"Go, go!"`, `3.14`, repeated words, Unicode, and standalone leading/interstitial/trailing punctuation.
  - Include exact ownership expectations for `“ Hello ! ”`, `Hello ( — world`, `Hello — world`, leading/trailing `!`, ellipses, em dashes, slashes, standalone ASCII `'`, and punctuation with no neighboring unit. Expect each atomic punctuation gap to belong to exactly one adjacent unit, every source punctuation character to appear exactly once, and reconciliation to fail when no unit can own it.
  - Expect segment count/order/timestamps to remain unchanged; missing, extra, reordered, and aligned-prefix-only units must raise a reconciliation error.
  - Test cue boundaries at 10/11 units, 42/43 characters, 6,000/6,001 ms, 799/800 ms gaps, sentence punctuation before closing quotes, and timestamp formatting across second/minute/hour carries.
  - Define caption comparison as `" ".join(text.split())`; assert concatenated cue text equals the canonical transcript under that normalization and SRT/VTT cue texts are identical.
- **Risk/recovery:** The source-ownership policy is deliberately English-only. A mismatch is an actionable hard failure; multilingual/CJK handling requires a separate future design rather than fallback heuristics.

#### T4 — Define exact artifact bytes, safe publishing, and reproducible acceptance

- **Change:**
  - Define `.words.json` as UTF-8, `ensure_ascii=False`, indentation of 2, and one trailing LF with the exact top-level keys `schema_version`, `text`, `language`, and `words`; `schema_version` is integer `1`, `language` is `English`, and every `words[]` item has `text`, `start_ms`, and `end_ms`.
  - Document that `words[]` maps 1:1 to forced-aligner units, not guaranteed linguistic words; each `text` includes its owned source punctuation, while root `text` is the only lossless canonical transcript.
  - Render TXT as canonical text plus exactly one final LF. For the accepted empty result, emit exact bytes: TXT `b"\n"`, JSON representing `{"schema_version": 1, "text": "", "language": "English", "words": []}` under the formatting rule, SRT `b""`, and VTT `b"WEBVTT\n\n"`.
  - Derive all paths from the input stem and fail before model resolution if any exists unless `--overwrite`; reject a target path that is a directory. Resolve/normalize the input and target paths and reject any target that aliases the input—including filesystem identity via `samefile` when both exist—regardless of `--overwrite`.
  - Render every payload in memory, then prepare same-directory temporary files. With `--overwrite`, publish each complete file using `os.replace`. Without it, use an atomic no-clobber commit (`os.link(temp, target)` followed by unlinking the temp) so a target created after preflight is never overwritten. Clean all remaining temporary files on every failure.
  - State the guarantee precisely: no target is ever visible with partial bytes, but a commit failure may leave a subset of complete artifacts because cross-file transactions are out of scope. Do not add `fsync` or claim crash durability.
  - Print the four completed paths to stdout, send progress/errors to stderr, and exit 0 only after all four commits succeed.
  - Add unit tests that inject write and commit failures, including a destination appearing after preflight; assert pre-existing/racing target bytes remain unchanged, any created artifact is complete, and no temporary files remain. Test lexical path equality and same-file aliases to ensure `--overwrite` can never replace the input.
  - Document installation, ffmpeg, the ~6.1 GiB model cache, cold download and cache-first warm behavior, `HF_HUB_OFFLINE=1`, output schema/bytes, overwrite semantics, English-only punctuation guarantee, and deferred features in `README.md`.
  - Add `scripts/validate_smoke.py` with this independently observed expected transcript for the pinned official sample: `Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.` The script must compare TXT/JSON root text to that literal oracle, completely parse TXT/JSON/SRT/VTT, validate the exact schema, monotonic integer timestamps, punctuation-preserving normalized caption equality, exact four-file set, and print `SMOKE OK`; it must not derive its oracle from generated outputs.
- **Starts at:** `src/stt/outputs.py` (`render_words_json`, `target_paths`, `preflight_targets`, `publish_artifacts`), `src/stt/cli.py`, `README.md`, `scripts/validate_smoke.py`, `tests/test_outputs.py`, `tests/test_cli.py`
- **Depends on:** T3
- **Verify:**
  - Run `uv run pytest -q`; expect all model-free unit/CLI tests to pass.
  - Run `uv run ruff check . && uv run ruff format --check .`; expect both commands to exit 0 with no changes.
  - Run `uv run stt /path/that/does-not-exist.wav`; expect nonzero exit, concise stderr, and no output artifacts.
  - Run this target-Mac smoke sequence; expect the named signals:

    ```bash
    set -euo pipefail
    tmp="$(mktemp -d /tmp/stt-smoke.XXXXXX)"
    trap 'rm -rf "$tmp"' EXIT
    audio="$tmp/asr_en.wav"
    out="$tmp/out"
    curl -L --fail --silent --show-error \
      -o "$audio" \
      https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav
    printf '%s  %s\n' \
      f9b4440ac8393e47c14a6240e9739dea09b645bb1592b8f2dd48feb9666cea7f \
      "$audio" | shasum -a 256 -c -

    uv run stt "$audio" -o "$out"
    uv run python scripts/validate_smoke.py "$out"
    find "$out" -maxdepth 1 -type f -exec shasum -a 256 {} + | sort > "$tmp/before.sha"

    if uv run stt "$audio" -o "$out"; then
      echo 'expected no-overwrite failure' >&2
      exit 1
    fi
    find "$out" -maxdepth 1 -type f -exec shasum -a 256 {} + | sort > "$tmp/after.sha"
    cmp "$tmp/before.sha" "$tmp/after.sha"

    HF_HUB_OFFLINE=1 uv run stt "$audio" -o "$out" --overwrite
    uv run python scripts/validate_smoke.py "$out"
    test "$(find "$out" -maxdepth 1 -type f | wc -l | tr -d ' ')" = 4
    test -z "$(find "$out" -maxdepth 1 -type f -name '.*.tmp-*' -print -quit)"
    ```

    Expect checksum `OK`, two `SMOKE OK` lines, the intentional second-run nonzero collision, unchanged pre/post hashes, successful `HF_HUB_OFFLINE=1` overwrite, exactly four artifacts, and no temporary files.
- **Risk/recovery:** The official fixture and immutable model/runtime make formatter acceptance reproducible. The expected transcript characterizes Qwen output; future intentional dependency/model updates must update this fixture expectation through an explicit review, not silent fallback.

## Final acceptance

- **Checks:**
  - `uv sync --frozen --dev`
  - `uv run pytest -q`
  - `uv run ruff check .`
  - `uv run ruff format --check .`
  - `uv run stt --help`
  - The complete M1 Max smoke block in T4, including pinned fixture checksum, exact validator, collision byte preservation, warm `HF_HUB_OFFLINE=1` execution, expected file count, and temporary-file cleanup.
- **End state:** `stt INPUT` performs one English Qwen3-ASR-1.7B transcription plus forced-alignment pass and produces punctuated TXT, schema-v1 punctuated timed-unit JSON, SRT, and VTT. Pinned runtime/model inputs are reproducible, warm use is demonstrably network-free, output collisions are explicit and race-safe, and invalid/incomplete alignment never silently loses punctuation.
- **Deferrals or blockers:** Multilingual support, batches, live/streaming input, speaker diarization, one-word subtitle files, caption policy flags, transcript rewriting, quantization, APIs/servers, cross-file transactions, crash durability, and air-gapped model distribution remain intentionally deferred. No implementation blocker remains.
