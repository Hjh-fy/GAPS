# Phase 3 Retry 1 Launch Failure Audit

- Status: `FAILED_CLOSED_BEFORE_SCRIPT_EXECUTION`
- Scope: launcher invocation only
- Recorded at: 2026-08-11 (Asia/Shanghai)

`Start-Process -ArgumentList` flattened an absolute `--output` value containing
spaces into multiple command-line tokens. Python therefore exited in
`argparse` with `unrecognized arguments` before the Phase 3 script executed.

Consequences:

- no preprocessing or protocol was changed;
- no classifier or R84 checkpoint was loaded or modified;
- no calibration fitting occurred;
- the sealed C5 test was not opened;
- no endpoint was completed and no result may be reused from this attempt.

The stderr log and PID are retained as provenance. A subsequent attempt must
use a new output directory and a repository-relative, space-free output
argument. No scientific configuration may change.
