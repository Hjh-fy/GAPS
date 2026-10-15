# Phase 3 Retry 2 Launch Failure Audit

- Status: `FAILED_CLOSED_BEFORE_SCRIPT_EXECUTION`
- Scope: Windows process launcher only
- Recorded at: 2026-08-11 (Asia/Shanghai)

The repository-relative output argument fixed the prior quoting issue, but
Windows PowerShell `Start-Process` failed before creating the child process
because its environment dictionary contained case-colliding `Path` and `PATH`
keys (`An item with the same key has already been added`).

Consequences:

- the Phase 3 Python entry point did not execute;
- no preprocessing, protocol, model, checkpoint, or hyperparameter changed;
- no calibration fitting occurred;
- the sealed C5 test was not opened;
- no endpoint was completed and no scientific result exists for this attempt.

The empty launcher artifacts are retained as provenance. The next attempt must
use a new output directory and direct foreground execution, without changing
the registered Phase 3 configuration.
