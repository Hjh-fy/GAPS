# C5 low-label commissioning analysis

Status: **LABEL_EFFICIENCY_NOT_SUPPORTED**.

## Frozen scope

This is the seed42 canonical-v1 C5 window-level commissioning label-budget sensitivity. It reuses the formal 20% target metrics and adds exactly six fresh 25-round endpoints. It does not establish few-shot unseen-experiment, strict cross-experiment, or deployment-independent generalization and does not weaken the existing strict C5 collapse finding.

## Primary result

- Budgets 20/15/10/5% use 320/240/160/80 calibration windows.
- All budgets cover 40/40 class × concentration strata, so performance changes are attributable primarily to quantity reduction rather than concentration-support loss.
- A0T Macro-F1, 20/15/10/5%: 99.4139%/99.4104%/98.8969%/97.3539%.
- GAPS/A4 Macro-F1, 20/15/10/5%: 99.4126%/98.6778%/97.8016%/96.7031%.
- GAPS/A4 minus A0T, 20/15/10/5%: -0.0013 pp/-0.7326 pp/-1.0953 pp/-0.6508 pp.

## Required answers

1. The complete A0T curve is reported above.
2. The complete GAPS/A4 curve is reported above.
3. Gap expansion is assessed directly in c5_budget_comparison.csv.
4. The first practically meaningful difference is the first 10% or 5% row with a prespecified one-percentage-point gap; absent such a row, no label-efficiency advantage is supported.
5. Practical significance uses the preregistered one-percentage-point gate; seed42 alone is not stability evidence.
6. The 5% subset covers all 40 strata.
7. Stratum coverage is therefore 40/40 (100%).
8. Any performance change is more consistent with label-quantity reduction than stratum-support loss under this nested family.
9. Source forgetting is reported relative to canonical FedAvg in c5_budget_source_retention.csv.
10. GAPS/A4 may be called more label-efficient only when status is LABEL_EFFICIENCY_SUPPORTED.
11. Main-text use requires claim wording consistent with the status and the strict-boundary limitation above.
12. Multi-seed recommendation: **NO**. No additional run was launched automatically.

## Stop rule

The six-run C5 classification study is complete. No C3/C4, lower-budget, multi-seed, R84, QC, method, preprocessing, or hyperparameter extension is authorized by this result.
