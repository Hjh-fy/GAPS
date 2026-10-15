# Experiment audit

- Verdict: AUDITED_SENSITIVITY_EVIDENCE.
- Frozen classifier, Federated H1, feature schema, alpha grid, and test universe verified.
- Filename groups remain intact within nested subsets and calibration-only folds.
- `calibration_fold_assignment_audit.csv` and `fold_isolation_audit.json` replay every low-budget fold from the frozen subset seed/algorithm and report zero filename leakage.
- Calibration lock existed and was SHA-bound before the one-shot low-calibration test stage.
- Historical calibration/test split remains window-level; original-file independence is not claimed.
- No QC, threshold, method, subset, alpha-grid, or budget was selected from test results.
