# Canonical S2/S4 FedRidge protocol proposal

Status: `PROPOSAL_ONLY_NOT_EXECUTED`.

Phase 4 stopped before fitting because the current frozen S2 H1 asset was
trained on historical 10 Hz / `100x8` windows. A fair canonical source-diversity
study must first reconstruct both S2 and S4 from the frozen 5 Hz / `50x8`
role-view dataset.

Proposed preregistered protocol:

- S2 clients: C1+C2; S4 clients: C1+C2+C3+C4; C5 is excluded from every source
  fit, source-alpha selection, refit, and sufficient statistic.
- H1 feature definition, scaler, explicit unregularized intercept, normal
  equations, clipping, and two-phase sufficient-statistics exchange remain
  identical to the registered H1 implementation.
- Apply the same source-only alpha grid independently to S2 and S4:
  `0, 0.01, 0.1, 1, 10, 100, 1000`.
- Fit candidates on each pool's source train split, select by aggregate
  source-calibration RMSE, and refit on source train+calibration. Source test and
  all C5 splits are unavailable to alpha selection.
- Lock both H1 manifests and prediction hashes before opening C5 test. Use the
  same frozen post-hoc C5 classifier, C5 calibration identities, target Ridge
  alphas, and scopes.
- Decision thresholds remain those in the task: >=5% S4 RMSE improvement,
  2--5% modest, <2% or inconsistent not supported. No further source-pool or
  alpha search follows.

Manual approval is required because this proposal reconstructs the currently
published S2 quantitative-prior asset; no execution occurred in this stage.
