# H8 + Formal C4 Guardrail Audit

- analysis_predictions: `results\formal_c4_route_rescue_selector_20260625\formal_c4_route_rescue_predictions.csv`
- selected_gate: `results\formal_c4_route_rescue_selector_20260625\formal_c4_route_rescue_selected_gate.json`
- equivalence_summary: `results\equivalence_h8_formal_c4_rescue_candidate_20260626\equivalence_summary.json`

## Gate Hits

- hit_N: 1
- hit_true_C4_high_CO_N: 1
- hit_false_N: 0
- hit_nonCO_N: 0
- C4_high_CO_recall: 0.00980392156862745
- guardrail_status: pass

## RMSE Before/After

| scope | N | RMSE before | RMSE after | delta |
| --- | ---: | ---: | ---: | ---: |
| C4_high_CO | 102 | 32.217 | 26.7913 | -5.42567 |
| C4_nonCO | 1020 | 8.85562 | 8.85562 | 0 |
| nonCO_ALL | 4050 | 18.3793 | 18.3793 | 0 |
| ALL | 5400 | 18.4686 | 18.3041 | -0.164455 |

## Interpretation

- `pass` requires zero false hits, zero nonCO hits, and runtime equivalence with zero mismatches when an equivalence summary is provided.
- H2.3 should remain the balanced default; this audit only supports H8+C4 as a CO/high-CO specialist candidate.
