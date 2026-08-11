# Final regression/QC closure report

## A. Final pipeline closure

- Source-only -> post-hoc classification: confirmed for C5 only.
- C3/C4 final post-hoc endpoints: unavailable.
- C5 classification: Macro-F1 0.976544; valid.
- C5 diagnostic R84 S_ALL: RMSE 28.057496 ppm, NRMSE 0.199727.
- HC90 diagnostic: 83.4559% coverage, RMSE 24.642191 ppm.
- HC95 diagnostic: 89.3382% coverage, RMSE 25.178852 ppm.
- Decision: `PIPELINE_CLOSURE_BLOCKED` because frozen H1 is legacy 100x8.

## B. FedRidge prior

- M83 diagnostic RMSE: 27.094018 ppm.
- M84 diagnostic RMSE: 28.057496 ppm.
- Delta RMSE: +0.963478 ppm; grouped 95% CI [-1.377391, 3.942463].
- C3/C4/pooled: blocked.
- Decision: `FEDRIDGE_PRIOR_NOT_SUPPORTED`; evidence also fails canonical provenance.

## C. QC

- Q3 NRMSE AURC: 0.099131.
- Q1 confidence NRMSE AURC: 0.039913.
- Q0 random NRMSE AURC: 0.099492.
- At actual HC90/HC95 retained counts, Q3 beats random but is worse than Q1.
- Diagnostic decision: `QC_RANKING_SUPPORTED__MULTISIGNAL_ADVANTAGE_NOT_SUPPORTED`.
- Final decision: blocked by upstream H1 provenance.

## D. Source-diversity FedRidge

- S2 canonical reconstruction: not executed.
- S4 canonical reconstruction: not executed.
- Decision: `SOURCE_DIVERSITY_QUANT_PRIOR_BLOCKED_PREPROCESSING`.

## E. Manuscript impact

- Contribution 2: `LOCAL_METHOD_REVISION_REQUIRED`.
- Contribution 3: `RESTRICT_EXISTING_CLAIM`.
- Exact affected sections are recorded in
  `MANUSCRIPT_REGRESSION_QC_IMPACT_AUDIT.md`.

## F. Final project decision

`PIPELINE_EVIDENCE_CONFLICT_REQUIRES_DISCUSSION`

No new training, extension, or algorithm search was started after the hard fail.
