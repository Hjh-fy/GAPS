# Task 1 Report: IoT-J Input Freeze

Date: 2026-07-11

## Status

DONE_WITH_CONCERNS. The read-only local audit, focused tests, and frozen manifest are complete. The clean matrix artifacts remain controller-owned and unavailable locally.

## Changed Files

- `scripts/audit_iotj_experiment_inputs.py`: read-only SHA-256 inventory, primary split summary, and required matrix-run gate.
- `tests/test_iotj_experiment_input_audit.py`: manifest-field, missing-matrix, and primary split-count coverage.
- `results/iotj_experiment_freeze_20260711/input_manifest.json`: generated immutable local input manifest.
- `docs/experiments/iotj_system_experiment_notebook.md`: Task 1 command, output, finding, failure, and next-action record.
- `.superpowers/sdd/task-1-report.md`: this report.

## TDD Evidence

1. Initial focused test command:

   ```text
   python -m pytest tests/test_iotj_experiment_input_audit.py -q
   ModuleNotFoundError: No module named 'scripts.audit_iotj_experiment_inputs'
   ```

   This was the expected red state before the audit module existed.

2. Protocol-freeze assertion added after the first green state:

   ```text
   ..F
   KeyError: 'calibration_fit_ratio'
   ```

   This was the expected red state before the manifest explicitly recorded the fixed `75%/25%` calibration fit/validation partition.

3. Final focused verification:

   ```text
   python -m pytest tests/test_iotj_experiment_input_audit.py -q --basetemp .tmp_pytest
   ...                                                                      [100%]
   3 passed in 0.21s
   ```

4. Compile check:

   ```text
   python -m py_compile scripts/audit_iotj_experiment_inputs.py
   Exit code: 0
   ```

The unmodified default pytest command cannot create its `tmp_path` fixtures because the system temporary base raises `WinError 5` before test execution. The focused suite passes with the workspace-local `--basetemp .tmp_pytest` override; no production behavior depends on that override.

## Generated Manifest

Command:

```text
python scripts/audit_iotj_experiment_inputs.py --output results/iotj_experiment_freeze_20260711/input_manifest.json
Wrote results\iotj_experiment_freeze_20260711\input_manifest.json: 109 artifacts, 6 matrix runs, status=incomplete
```

Summary:

- Primary dataset status: `complete`; data root is `dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`.
- Protocol freeze: source clients `C1,C2`; target clients `C3,C4,C5`; seed `42`; calibration/test `20%/80%`; inner calibration fit/validation `75%/25%`.
- Target calibration/test counts: C3 `680/2680`, C4 `320/1360`, C5 `320/1360`.
- Artifact records: 97 present and 12 missing, each with resolved path, role, existence, byte size, SHA-256, and status.
- Matrix inventory: F4, F5, R1, R2, R3, and R4 are each `missing` both `server_latest_adapted.pth` and `run_config.json`.

## Self-Review

- Confirmed the script uses chunked `hashlib.sha256` reads and writes only the selected manifest output.
- Confirmed unavailable matrix files remain explicit manifest records instead of removing unrelated local inputs.
- Confirmed primary split counts derive from saved feature, classification-label, and regression-label arrays without changing them.
- Reviewed the staged task-only diff and ran `git diff --cached --check` successfully.
- Verified the generated JSON parses, reports a complete primary dataset, preserves the calibration partition, and reports exactly six missing priority matrix runs.

## Concerns and Next Action

The manifest remains `incomplete` solely because the controller has not copied `/root/GAPS/results/source_target_classification_matrix_20260708_clean/` to `results/source_target_classification_matrix_20260708_clean/`. No SSH/SCP or other cloud transfer was attempted. After controller recovery, rerun the manifest command to freeze the recovered checkpoints and configs.

## Commit

Task implementation commit: `7d3213a` (`feat: freeze IoT-J experiment inputs`).
