# C5/H8 Runtime Contract B5 v3 Superseded Audit

Status: `superseded` / not valid as formal runtime-parity evidence.

The v3 contract corrected the v2 dataset-root mismatch and produced a valid 1,360-row probability-signature bijection. However, it did not bind `test_phase_labels.npy`, while the frozen H2.3/R4 feature schema requires `phase_id_0`, `phase_id_1`, `phase_id_2`, and `phase_id_unknown`.

Fail-closed runtime execution must not infer the numeric phase from a metadata label. The non-overwriting v4 contract therefore adds the phase-label file as a required SHA-256-bound input and records the source `float64` to runtime `float32` cast explicitly.

No HC95 or HC90 success report was issued from v3. The sole formal candidate for this closure is:

- `c5_h8_runtime_contract_b5_v4/runtime_contract.json`
- `c5_h8_runtime_contract_b5_v4/row_map_1360.json`

This correction does not train, refit, or alter any frozen B5, H2.3, R4, risk-calibrator, or QC-policy asset.
