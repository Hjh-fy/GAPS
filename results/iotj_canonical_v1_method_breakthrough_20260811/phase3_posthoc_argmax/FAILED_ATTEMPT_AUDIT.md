# Phase-3 failed attempt audit

- Status: `FAIL_CLOSED_BEFORE_CALIBRATION_FIT`.
- Freeze commit: `b312c0e`.
- Selected classifier: immutable I0+B20 post-hoc step100 checkpoint.
- Failure boundary: calibration classification routing, before R84 model fitting, calibration lock creation, or target-test opening.
- Root cause: the reused canonical `route_rows` helper implicitly applied its historical default endpoint contract `round=25`; the selected post-hoc classifier correctly records `step=100`.
- No model, prediction, calibration lock, or test result was produced in this directory.
- Resolution: add an explicit `expected_endpoint` parameter while preserving the round25 default for all historical callers. The retry must use a new `retry1/` output directory.

