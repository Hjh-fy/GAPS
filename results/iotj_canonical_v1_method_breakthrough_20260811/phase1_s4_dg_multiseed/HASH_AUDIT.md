# Phase-1 hash audit

## Verdict: PASS WITH PRESERVED WRAPPER-INDEX DEFECT

- The runner-generated `sha256_index.json` contained 429 entries.
- Exactly one entry differed: `RUN_PROGRESS.json`, because the wrapper changed its status from endpoint-locked to complete after creating the original index.
- The other 428 immutable experiment files matched their recorded SHA256 values.
- The original index is retained unchanged as provenance of the generation-order defect.
- `sha256_index_final.json` is the authoritative post-run integrity index. It explicitly excludes mutable wrapper files (`RUN_PROGRESS.json`, PID, stdout, and stderr) and verifies 428 immutable files with zero failures.
- No checkpoint, prediction, metric, split, or protocol file was replaced or recomputed to resolve this defect.

