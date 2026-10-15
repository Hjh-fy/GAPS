# GAPS canonical-v1 final regression/QC evidence closure

Status: frozen for execution at classification-method freeze commit `d950062`.

## Scientific objective

Close the quantitative-regression and frozen-QC evidence chain for the final
post-hoc IoT commissioning lifecycle. This stage does not reopen classification
method exploration and does not modify canonical preprocessing, target splits,
R84, Ridge alphas, or the QC formula.

## Frozen design

- Dataset: `canonical-v1`; preprocessing `HZ5_MEAN_W10S`, 5 Hz, 10 s,
  stride 5 s, input `50x8`.
- Source devices: C1+C2. Targets: C3/C4/C5; primary target: C5.
- Calibration/test: frozen canonical manifests; target test remains a final
  evaluation split and is never used for model, alpha, threshold, or endpoint
  selection.
- Classifier: final post-hoc lifecycle endpoint selected before this closure.
- Regression: `R84_FED_H1`, fixed per-gas Ridge alphas, seed 42.
- QC: `equal_mean_of_calibration_p95_normalized_components`; calibration-only
  component scaling and quantile threshold locking; no test-based retuning.

## Execution order and stopping rules

1. Phase 0 audits provenance, state-content fingerprints, calibration identity,
   split sealing, and lifecycle consistency before any derived analysis.
2. Phase 1 reuses the completed C5 post-hoc fixed endpoint and constructs a
   lifecycle-consistent C5 QC lock from calibration-only risk fields. No
   classifier or regressor is retrained. C3/C4 are proposals because no formal
   final post-hoc endpoint exists for those targets.
3. Phase 2 compares M83 and M84 on C5 with grouped bootstrap by highest raw
   experiment/file identity. Cross-target support remains blocked until C3/C4
   post-hoc endpoints exist.
4. Phase 3 evaluates Q0--Q3 at the registered coverage grid and at the actual
   HC90/HC95 retained counts, using the same Phase-1 predictions.
5. Phase 4 evaluates S2 versus S4 FedRidge only if an auditable source-only S4
   alpha protocol already exists. Otherwise it stops before evaluation and
   emits a protocol proposal.
6. Phase 5 is proposal-only. Phase 6 audits manuscript claim impact without
   editing the manuscript.

Any provenance mismatch, target-test selection, missing endpoint, altered
hyperparameter, or identity-count mismatch is a hard fail for the affected
claim. Missing C3/C4 final post-hoc endpoints must not be replaced with the
historical interleaved A4 endpoints.
