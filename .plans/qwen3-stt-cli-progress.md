# Implementation Progress

- **Template loaded from:** `implement-plan/assets/progress-tracker-template.md`
- **Plan:** `.plans/qwen3-stt-cli.md`
- **Status:** `Complete`
- **Updated:** 2026-07-24

`Complete` = all rows `Verified` or user-approved `Descoped` + validation passed + final review `Clear` + nothing material open.

Parent = sole tracker writer under concurrency.

## Tasks / subtasks

Status: `Pending` | `In progress` | `Blocked` | `Verified` | `Descoped`

| ID | Plan ref / requirement | Deps | Status | Acceptance check | Evidence |
|---|---|---|---|---|---|
| T1 | Scaffold pinned package + CLI contract | — | Verified | `uv sync --frozen --dev`; `uv run stt --help`; `uv run pytest -q tests/test_cli.py` | Parent: sync frozen OK; help contract OK; 6 CLI tests; mlx 0.3.5 |
| T1.1 | `pyproject.toml` src-layout, Python `>=3.12,<3.13`, pin `mlx-qwen3-asr==0.3.5`, `huggingface-hub>=0.20,<2`, pytest/ruff dev deps, console `stt = "stt.cli:main"`, commit `uv.lock` | — | Verified | lock frozen; package installable | pyproject + uv.lock |
| T1.2 | `.gitignore` for `.venv`, caches, generated outputs, optional model caches; keep fixtures/test data | T1.1 | Verified | ignores correct paths | no blanket `*.txt` |
| T1.3 | `src/stt/__init__.py` + `cli.py` argparse contract | T1.1 | Verified | help + path validation tests | cli.py |
| T1.4 | `tests/test_cli.py` model-free CLI tests | T1.3 | Verified | pytest passes | included in 111 |
| T2 | Resolve pinned models cache-first + validate one inference result | T1 | Verified | `uv run pytest -q tests/test_engine.py`; version assert 0.3.5 | engine + preflight |
| T2.1 | Model IDs + immutable revisions; cache-first snapshot; online retry only on miss; `HF_HUB_OFFLINE=1` | T1 | Verified | cache-first/online-retry tests | engine.py |
| T2.2 | Collision preflight before model resolution; one `transcribe(...)` call | T2.1 | Verified | one-call + exact args tests | preflight before resolve |
| T2.3 | Frozen local ms records + validation accept/reject matrix | T2.2 | Verified | characterization tests | normalize_result |
| T2.4 | Wire engine into CLI; concise nonzero failures | T2.3 | Verified | CLI failure path | orchestration |
| T3 | Reconcile canonical English source spans + generate cues | T2 | Verified | `uv run pytest -q tests/test_captions.py` | 50 caption tests |
| T3.1 | `clean_alignment_token` | T2 | Verified | cleaner unit tests | captions.py |
| T3.2 | `restore_source_units` ownership policy | T3.1 | Verified | punctuation/ownership/mismatch tests | |
| T3.3 | `group_cues` fixed v1 limits | T3.2 | Verified | boundary tests | |
| T3.4 | `render_srt` / `render_vtt` | T3.3 | Verified | format + equality tests | |
| T4 | Exact artifact bytes, safe publishing, reproducible acceptance | T3 | Verified | full pytest; ruff; smoke block | all green |
| T4.1 | `outputs.py` schema-v1, empty bytes, atomic publish | T3 | Verified | `tests/test_outputs.py` | |
| T4.2 | Wire publish into CLI; stdout paths; exit 0 after all 4 | T4.1 | Verified | CLI + smoke | |
| T4.3 | `README.md` | T4.1 | Verified | docs present | README.md |
| T4.4 | `scripts/validate_smoke.py` | T4.1 | Verified | SMOKE OK ×2 on real model | |
| T4.5 | Final acceptance suite | T4.1–T4.4 | Verified | pytest 111; ruff; help; full M1 smoke | parent-run smoke block complete |

## Loop log (optional, keep brief)

| ID | Owner | Worktree / isolation | Checks | Review | Cleanup |
|---|---|---|---|---|---|
| T1 | worker run-mryxrjhx-500281 | shared checkout | parent: 6 passed, help, frozen sync | parent spot-check Clear | stopped |
| T2 | worker run-mryxuk05-591544 | shared checkout | parent: 38 passed | parent spot-check Clear | stopped |
| T3 | worker run-mryxzuz1-78bbaf | shared checkout | parent: 88 passed, ruff clean | parent spot-check Clear | stopped |
| T4 | worker run-mryy5vml-aa60e8 | shared checkout | parent: 111 passed; ruff; live smoke OK | final plan-backed Clear | stopped |
| Final review | research run-mryyefkd-291491 | shared checkout (RO) | matrix + 4 verdicts Clear | no material findings | stopped |

## Reviews

| Checkpoint | Reviewer | Findings | Disposition | Closure |
|---|---|---|---|---|
| Final full-plan | research run-mryyefkd-291491 | none | — | Clear |
| Parent acceptance | parent | none | — | Clear |

## Decisions / deviations

| Item | Need / change | Evidence | Status |
|---|---|---|---|
| Incomplete-prefix reject in captions (not engine normalize) | Still all-or-nothing before publish | captions.restore_source_units | Accepted |
| ReconciliationError wrapped as PublishError | Message preserved; nonzero exit | outputs.publish_artifacts | Accepted |
| Aligner keep-set duplicated engine/captions | Identical keep rules | engine + captions | Accepted residual |
