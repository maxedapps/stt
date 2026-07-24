# Decomplex review: English Qwen3 speech-to-text CLI plan

## Overall status

No potential complexity findings.

## Review contract

| Axis | Selection |
|---|---|
| Mode | Prevention |
| Target | `.plans/qwen3-stt-cli.md` |
| Authority / required behavior | English-only local Apple-Silicon Qwen3 CLI producing punctuated TXT, word JSON, SRT, and VTT |
| Scope | Planned source boundaries, dependencies, model resolution, validation, output safety, schema, CLI flags, and tests |
| Report | `.reviews/qwen3-stt-cli-decomplex.md` |

## Coverage

### Inspected

- Four-module src layout and task boundaries.
- Pinned MLX runtime and immutable, cache-first Hugging Face model resolution.
- One-input CLI and intentionally omitted feature controls.
- Exact source-span punctuation reconciliation and fail-closed behavior.
- Word JSON schema and local SRT/VTT rendering.
- Timestamp/result validation, empty-result semantics, race-safe no-clobber/overwrite behavior, and per-file atomic visibility.
- Model-free unit/CLI/lint checks and one reproducible real-model smoke validator.

### Skipped or partial

- No implementation exists, so source-level burden and measured runtime behavior cannot yet be audited.
- Multilingual, batch, streaming, diarization, servers, quantization, and air-gapped packaging are explicit non-goals.

## Potential findings

**No potential complexity findings.**

## Confirmed proportionate areas

- A narrow engine boundary is justified by the pre-1.0 third-party runtime and immutable model snapshots.
- Local punctuation reconciliation is required because the forced aligner irreversibly strips punctuation from timed units.
- Separate caption and output modules keep linguistic mapping independent from filesystem/schema behavior without introducing a framework or registry.
- Exact matching and fail-closed behavior are smaller and safer than fuzzy token alignment or fallback chains.
- Cache-first immutable snapshot resolution is justified by the required warm local behavior and avoids a separate download manager.
- Per-file temporary writes and atomic no-clobber commits address reachable overwrite races without claiming or implementing cross-file transactions or crash durability.
- Fixed English/model/output behavior and three CLI arguments avoid speculative configuration.
- Model-free unit tests plus one small smoke validator for the pinned official sample avoid both a heavy mandatory model test lane and unverified integration.

## Limitations

- The report assesses planned complexity only; implementation may introduce avoidable layers or duplication not visible yet.
