# Split compatibility audit

## Verdict

The existing `FCL-E3-GAPS-C3/C4/C5` target-adapted checkpoints are
**ineligible** for the corrected role-aware test.

## Evidence

- Existing FCL-E3 runs used
  `client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`.
- The corrected study uses
  `client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`.
- Only the latter declares C3, C4, and C5 jointly as targets with target train
  disabled and a registered 20/80 calibration/test split.
- Legacy experiment metadata omits exact window timestamps, so exact
  checkpoint-to-new-test membership cannot be proven from stable IDs.
- A same-filename nearest-window diagnostic maps 257/320 C3, 131/160 C4,
  and 253/320 C5 legacy calibration windows to the new role-aware test. This
  approximately 80% pattern is consistent with two different window-level
  splits and is sufficient to fail closed; the diagnostic is not treated as
  an exact identity count.

## Resolution

Run three new target-specific GAPS branches with the frozen FCL-E3 algorithm
and the corrected role-aware target calibration. Target tests remain closed
during all 25 rounds and server adaptation. Existing checkpoints and results
remain read-only and are retained as legacy-split diagnostics.
