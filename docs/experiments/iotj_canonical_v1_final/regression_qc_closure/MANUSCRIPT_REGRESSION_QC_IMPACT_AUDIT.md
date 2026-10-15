# Manuscript regression/QC impact audit

No manuscript text was edited.

| Section | Impact | Required action |
|---|---|---|
| Abstract | RESTRICT_EXISTING_CLAIM | Remove final-canonical quantitative superiority until canonical H1 is reconstructed. |
| Contribution 2 | LOCAL_METHOD_REVISION_REQUIRED | Distinguish the historical 100x8 H1 evidence from the canonical 50x8 lifecycle. |
| Section 4.3 FedRidge | LOCAL_METHOD_REVISION_REQUIRED | State the preprocessing provenance of the existing H1 asset; do not label it canonical-v1. |
| Section 4.4 target personalization | RESTRICT_EXISTING_CLAIM | Current C5 diagnostic shows M84 S_ALL is not better than M83; grouped CI crosses zero. |
| Section 4.5 QC | RESTRICT_EXISTING_CLAIM | Q3 beats random but not confidence-only; additionally its upstream R84 provenance is blocked. |
| Section 5 regression results | MAJOR_STORY_CONFLICT | Existing canonical final R84 values depend on a legacy H1 prior. Replace only after an approved canonical H1 rerun. |
| Section 5 QC results | MAJOR_STORY_CONFLICT | Treat current results as historical/diagnostic, not final canonical evidence. |
| Conclusion | RESTRICT_EXISTING_CLAIM | Retain lifecycle-framework language; suspend canonical algorithmic superiority. |

## Numerical conflicts

| Item | Old draft/historical value | New audit status | Reason | Classification |
|---|---:|---|---|---|
| C5 R84 S_ALL RMSE | 28.0575 ppm post-hoc diagnostic; earlier 12.855 ppm historical A4 package | not final canonical evidence | legacy 100x8 H1 mixed with canonical 50x8 target pipeline and different classifier lifecycle | historical/noncanonical |
| C5 M84 vs M83 | claimed prior benefit | Delta RMSE +0.9635 ppm; CI [-1.3774, 3.9425] diagnostic only | no C5 S_ALL support even before provenance failure | claim unsupported |
| QC equal mean | multi-signal mechanism implied | Q3 NRMSE AURC 0.09913 vs confidence-only 0.03991 diagnostic | multi-signal advantage absent and upstream regression blocked | restrict |

No rerun is authorized automatically. A canonical S2 H1 reconstruction is
required before any final regression/QC manuscript numbers can be approved.
