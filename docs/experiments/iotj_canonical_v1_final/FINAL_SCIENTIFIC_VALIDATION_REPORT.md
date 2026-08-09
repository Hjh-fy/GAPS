# Final canonical-v1 scientific validation report

Final recommendation: **NOT_READY**

| Claim | Evidence | Protocol | Canonical? | Result | Risk | Status |
|---|---|---|---|---|---|---|
| Ordinary source-only FL is insufficient for the observed target shift | Canonical FedAvg/FedProx/SCAFFOLD versus GAPS/A4 | same canonical data/backbone/25 rounds/LE1/seed42; optimizer regimes disclosed | yes | FedAvg GAPS-minus-baseline Macro-F1 C3/C4/C5=0.0593/0.0766/0.6255; FedProx GAPS-minus-baseline Macro-F1 C3/C4/C5=0.0394/0.0805/0.6010; SCAFFOLD GAPS-minus-baseline Macro-F1 C3/C4/C5=0.3491/0.3626/0.2785 | Different target-information and SCAFFOLD optimizer regimes; not a single-factor ablation | PASS_WITH_LIMITATION |
| Canonical equal-label A0T exists | Three fixed round-25 A0T endpoints and sealed evaluation | same calibration identities/label budget; target CE only | yes | C3/C4/C5 all complete | Single seed | PASS |
| Structured commissioning adds value beyond label access | GAPS/A4 minus equal-label A0T | matched canonical data, target label budget, backbone and fixed optimization budget | yes | Macro-F1 delta C3/C4/C5=0.000001/0.002937/-0.000013 | Near-zero mixed-sign single-seed deltas do not support material classification superiority beyond label access; GAPS value must be framed at lifecycle level | PASS_WITH_LIMITATION |
| Routing error propagates into regression | S_ALL, S_CC, oracle-route prediction-level analysis | canonical A4+R84 | yes | Overall S_ALL-S_CC RMSE gap 2.500 ppm; 15 misroutes | S_CC and oracle are different diagnostic populations, so evidence is descriptive rather than causal | PASS_WITH_LIMITATION |
| Federated H1 prior contributes to target regression | 83D/84D paired raw-file grouped bootstrap | 5000 replicates, seed20260809 | yes | Delta RMSE=-0.535 ppm, 95% CI [-1.458, 0.426] | CI includes zero; C4 degrades | PASS_WITH_LIMITATION |
| QC identifies higher-risk predictions | HC90/HC95, same-coverage random, capture and risk-coverage analyses | frozen equal-mean QC; no threshold search | yes | HC90/HC95 RMSE gain versus random=1.125/1.038 ppm | Misroute capture is 26.7%; risk evidence is useful but not exhaustive | PASS |
| Main conclusions survive strict non-overlap | raw-file-disjoint A4+R84 sensitivity | exact/raw-file/raw-time overlap all zero | supplementary canonical sensitivity | collapse flag=True | C5 triggers both the preregistered classification and regression collapse flags; C4/C5 test N is 840 after two-repeat grouping | BLOCKED |
| Low-budget/few-shot commissioning | No canonical budget sensitivity | predefined approximately 20% calibration only | yes | Claim restricted | Few-shot/limited-calibration wording unsupported | PASS_WITH_LIMITATION |
| Pi 5 evidence matches canonical deployed package | package hash chain and 10,000-window benchmark | FINAL_DEPLOYED_RUNTIME | yes | P50/P95/P99 3.149/3.193/4.924 ms; 295.93 windows/s | FL communication is analytical/historical, not canonical wire measurement | PASS |
| Additional algorithm exploration is required | Minimal comparator matrix plus fairness and robustness closure | stop rule | yes | No | None beyond disclosed limitations | NOT_REQUIRED |

## Decision-gate answers

1. Ordinary source-only FL inadequacy: supported for these canonical fixed endpoints, with supervision/optimizer limitations disclosed.
2. Canonical equal-label A0T: PASS.
3. GAPS versus A0T claim strength: classification performance is effectively tied under equal label access (mixed-sign, near-zero deltas at seed42); any added-value claim must be lifecycle-level, not classification-superiority wording.
4. Routing-to-regression propagation: supported descriptively by the S_ALL-S_CC gap and misroute slices.
5. 83D to 84D: modest overall average benefit; not statistically significant at the grouped-bootstrap 95% interval; retain C4 degradation.
6. QC versus same-coverage random: positive at HC90 and HC95; capture analysis shows it does not catch every misroute.
7. Strict non-overlap: BLOCKED because C5 triggers both preregistered collapse flags; canonical window-level evidence does not establish strict robustness.
8. Calibration-budget wording: only 'a predefined approximately 20% commissioning calibration set'; no few-shot claim.
9. Pi 5: package hash chain is consistent with the canonical A4+R84+QC deployment.
10. New algorithms: not required; stop algorithm exploration.

This report does not authorize manuscript-number edits, figure regeneration, model changes, hyperparameter search, outlier deletion, or additional algorithms.
