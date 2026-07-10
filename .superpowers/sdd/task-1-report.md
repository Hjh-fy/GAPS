# Task 1 Report: Corrected C12-to-C5 Input Freeze

Date: 2026-07-11

## Status

DONE. The corrected C1/C2 source and C5-only target inputs are validated and frozen. No local simulated training was run.

## Owned Files

- `scripts/audit_iotj_experiment_inputs.py`
- `tests/test_iotj_experiment_input_audit.py`
- `results/iotj_experiment_freeze_20260711/input_manifest.json`
- `docs/experiments/iotj_system_experiment_notebook.md`
- `.superpowers/sdd/task-1-report.md`

## Reviewer Fixes

1. Replaced matrix-root name heuristics with `audit_matrix_run(run_dir, expected)`. Arbitrary directory names receive required-file and config validation.
2. Switched dataset authority to structured `split_info.json`. Completion now rejects missing metadata, wrong C5 target, wrong seed, wrong ratios, target training, wrong stratification, missing active splits, and wrong C5 counts.
3. Made canonical F2 the only required run. Sibling directories are `inventory_only` and cannot change primary completeness.
4. Added protected-output validation. The output cannot equal an input artifact or be contained by the dataset or run directory.
5. Made `build_manifest` deterministic. The wall-clock value is isolated under `provenance.generated_at_utc` by `with_provenance`.
6. Restricted primary dataset summaries and artifacts to C1, C2, and C5. C3/C4 remain shared-root source arrays but are absent from primary clients, target rows, and artifact paths.

## TDD Evidence

Initial red run against the old implementation:

```text
python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest/task1-c12-c5
.FFFFFFFFFFFFFFFFFFFFFFFFFFF                                             [100%]
27 failed, 1 passed in 2.34s
```

The failures covered missing explicit run APIs, missing metadata/config validation, old C345 client selection, extra-run gating, output collisions, and nondeterministic payload structure.

Green run after implementation:

```text
python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest/task1-c12-c5
............................                                             [100%]
28 passed in 3.90s
```

## Generated Manifest

Command and exact output:

```text
python scripts/audit_iotj_experiment_inputs.py --data-root dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid --run-dir results/source_target_classification_matrix_20260630/F2_C12_to_C5_fixed_da_strong_r25 --output results/iotj_experiment_freeze_20260711/input_manifest.json
Wrote results\iotj_experiment_freeze_20260711\input_manifest.json: 228 artifacts, 1 required run, status=complete
```

Manifest summary:

- Overall status: `complete`.
- Artifacts: 228 present, 0 missing, 228 valid 64-character SHA-256 hashes.
- Active clients: exactly C1, C2, and C5; inactive C3/C4 artifact paths: 0.
- Dataset validation: `complete`, 0 errors, 1 warning.
- Warning: `split_info.protocol` is stale; structured target and split fields are authoritative.
- C5 calibration/test: 320/1360 windows.
- C5 calibration classes: 80/80/80/80; test classes: 340/340/340/340.
- Required runs: exactly one canonical F2 row, `complete`.
- F2 inventory: 183 files, 11,522,104 bytes; required config, history, and adapted checkpoint present.
- F2 config: rounds 25, strategy `gaps`, profile `strong_cls`, source validation C1/C2 only, target calibration C5 only, DA enabled, 100 steps, target CE 0, adapted weights used globally.
- Extra required runs: none. F5 and reverse R1-R4 do not participate in completeness.

## Self-Review Checklist

- [x] Read the corrected brief, current plan, and current experiment notebook before editing.
- [x] Preserved the visibly superseded C12-to-C345 notebook findings.
- [x] Kept `audit_inputs(paths)` for generic file hashing without matrix-name inference.
- [x] Tested arbitrary-name run validation and all three required F2 files.
- [x] Tested every structured dataset contract and exact C5 counts.
- [x] Tested every required F2 config field, including rejection of C3/C4 target calibration.
- [x] Tested that extra run directories are inventory-only.
- [x] Tested output equality and containment before writing.
- [x] Tested stable payload equality with provenance timestamps isolated.
- [x] Verified every present manifest artifact has a SHA-256 hash.
- [x] Verified no C3/C4 primary artifact paths or client summaries exist.
- [x] Ran no local simulated training, SSH, or SCP.
- [x] Reviewed only Task 1 owned-file diffs and ran whitespace checks.

## Concerns

No blocking concerns. The single validation warning is intentional: the free-text `split_info.protocol` label is stale, while all binding structured fields pass.

## Commit

Implementation commit: `719f353` (`fix: freeze corrected C12-to-C5 inputs`).
