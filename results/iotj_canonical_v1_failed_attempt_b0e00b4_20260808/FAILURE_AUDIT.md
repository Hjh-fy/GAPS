# Canonical v1 First-Start Failure Audit

Status: `FAIL_CLOSED_BEFORE_TRAINING`

Freeze commit: `b0e00b47604c256d34caa249bfa6aaf330cf965f`

The C3 server stopped during strategy construction before Flower clients trained. The source calibration array had the correct frozen canonical shape `(320, 50, 8)`, but the historical server DA guard used its default expected shape `(100, 8)`. The server raised a `ValueError`; both clients had empty logs, no round completed, no checkpoint was produced, and target test remained sealed.

The corrective patch adds an explicit `--da-window-length 50` declaration to the canonical command, passes `(50, 8)` to the existing DA array validator, and leaves the historical default at 100. It does not change preprocessing, array contents, model parameters, losses, optimizer, training budget, or target-information policy. Related tests: 70 passed.

The original local logs and locked run specification are retained under `classification/CANONICAL-V1-GAPS-C3/`. The corresponding ECS output was preserved as `failed_attempt_b0e00b4_CANONICAL-V1-GAPS-C3` rather than overwritten.
