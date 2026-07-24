# Final plan-backed review — qwen3-stt-cli

- **Plan:** `.plans/qwen3-stt-cli.md`
- **Tracker:** `.plans/qwen3-stt-cli-progress.md`
- **Date:** 2026-07-24
- **Reviewer:** independent research subagent + parent disposition
- **State:** `Clear`

## Verdicts

| Dimension | Verdict |
|---|---|
| Baseline readiness | Clear |
| Plan compliance | Clear |
| Quality beyond baseline | Clear |
| Tests / validation | Clear |

## Material findings

None.

## Validation evidence

- `uv sync --frozen --dev` OK
- `uv run pytest -q` → **111 passed**
- `uv run ruff check .` + `ruff format --check .` clean
- `uv run stt --help` contract OK
- Full M1 smoke block: fixture checksum OK, two `SMOKE OK`, no-overwrite collision preserved, `HF_HUB_OFFLINE=1` overwrite OK, exactly 4 artifacts, no temps

## Accepted deviations

1. Incomplete aligned-prefix rejection lives in `captions.restore_source_units` (still blocks publish).
2. `ReconciliationError` wrapped as `PublishError` with message preserved.
3. Aligner keep-set helpers duplicated in engine/captions (identical).

## End state

Met: `stt INPUT` runs one English Qwen3-ASR-1.7B + forced-align pass and writes punctuated TXT, schema-v1 timed JSON, SRT, and VTT with pinned reproducible models, cache-first warm offline use, race-safe publish, and hard-fail reconciliation.
