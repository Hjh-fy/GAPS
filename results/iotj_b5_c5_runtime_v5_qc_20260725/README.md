# IoT-J Runtime v5 QC closure

Formal decision: `RUNTIME_V5_QC_VALID_BUT_NOT_SUPERIOR`.

The selected risk is QC2: B5 confidence plus B5 representation
prototype/support distance.  QC3 did not pass the preregistered OOF risk-direction
gate, so Federated-H1-to-target-Ridge consistency was not included in the deployed
risk.  The immutable selection lock is `qc_selection_lock.json`, SHA-256
`64877b7676bc4497074a2619282e0d2fedd658db76c2d1868b91699edba6d518`.

## Main evidence

- `protocol_manifest.json`: amendment-v2 protocol and evidence boundary.
- `qc_calibration_fold_manifest.json`: deterministic 80-filename, five-fold map.
- `qc_calibration_oof_rows.csv`: 320 OOF rows; large row artifact, SHA-indexed.
- `qc_candidate_calibration_summary.csv`: QC1/QC2/QC3 OOF comparison.
- `qc_candidate_selection.json`: QC2 selection before test opening.
- `runtime_v5_qc_policy.json`: locked full-calibration policy; large runtime asset.
- `hc95_test_rows.csv`, `hc90_test_rows.csv`: frozen 1360-row decisions.
- `qc_test_summary.csv`, `qc_per_gas_summary.csv`,
  `qc_risk_decile_summary.csv`: descriptive test evaluation.
- `comparison_vs_runtime_v4.json`: automatically loaded v4 guard comparison.
- `runtime_qc_calibration_parity_report.json`,
  `runtime_qc_test_parity_report.json`: zero-mismatch runtime parity.
- `decision_gate.json`: final Q10 decision.
- `test_stage_failure_001.json`: preserved non-selection v4-adapter failure and
  constrained resume record.

Runtime v4 remains the formal deployment baseline.  No threshold or risk component
was changed after test opening.  Filename grouping applies only to calibration OOF;
the historical calibration/test split remains window-level and is not
original-file independent.
