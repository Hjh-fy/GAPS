# Number Consistency Audit — IoT-J protocol closeout

## Canonical sources

- Legacy component metrics:
  `results/iotj_minimal_gap_audit_20260726/component_ablation_inventory.csv`
- Formal final-B5 seed42 metric:
  `results/iotj_b5_multiseed_20260724/seed42_reference/classification_evaluation/seed42_classification_metrics.json`
- Frozen manuscript baseline:
  `docs/paper/GAPS_IoTJ_evidence_frozen_20260726.zh.html`
- New occurrence:
  `docs/paper/tables/table_legacy_classification_ablation_protocol_closed_20260726.csv`

## Occurrences and discrepancies

| Finding ID | Metric identity | Canonical value | Observed value | Location | Scope/unit/precision | Classification | Severity | Proposed correction | Source |
|---|---|---:|---:|---|---|---|---|---|---|
| N01 | A0 C5 test accuracy, seed42 | 0.2654411765 | 0.265441 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N02 | A0T C5 test accuracy, seed42 | 0.9823529412 | 0.982353 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N03 | A5 C5 test accuracy, seed42 | 0.7301470588 | 0.730147 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N04 | A6 C5 test accuracy, seed42 | 0.9801470588 | 0.980147 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N05 | B1 C5 test accuracy, seed42 | 0.9875000000 | 0.987500 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N06 | B2 C5 test accuracy, seed42 | 0.9926470588 | 0.992647 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N07 | B3 C5 test accuracy, seed42 | 0.9889705882 | 0.988971 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N08 | B4 C5 test accuracy, seed42 | 0.9897058824 | 0.989706 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N09 | B5 v3-screen C5 test accuracy, seed42 | 0.9889705882 | 0.988971 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | component audit |
| N10 | Final canonical B5 C5 test accuracy, seed42 | 0.9801470588 | 0.980147 | legacy table | N=1360, fraction, 6 decimals | compatible rounding | informational | none | frozen seed42 evaluation |

## Compatible rounding cases

All ten added Accuracy values match the canonical source after rounding to six
decimal places. Accuracy is retained as a fraction rather than converted to
percent. The v3-screen B5 and final canonical B5 remain separate metric
identities and are not merged.

## Unknown, conflict, stale, or missing-source cases

None in the added legacy table. A0/A0T/A5/A6 and B1–B5 retain their legacy
evidence labels, so their values are not silently promoted to canonical
final-B5 component effects.

## Blocking verdict and revision handoff

- Blocking mismatches: 0
- Major mismatches: 0
- Minor mismatches: 0
- Compatible rounding cases: 10
- Verdict: `PASS_NO_NUMERIC_CHANGE_TO_FROZEN_RESULTS`

The protocol-closed manuscript may proceed to narrative revision and
translation. No experiment or test reopening is required.
