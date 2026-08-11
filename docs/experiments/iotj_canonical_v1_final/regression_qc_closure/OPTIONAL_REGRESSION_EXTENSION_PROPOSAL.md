# Optional regression extension proposal

Status: `PROPOSAL_ONLY_NOT_EXECUTED`.

No extension is scientifically interpretable until the canonical S2 H1 asset is
reconstructed and the base R84/QC chain passes provenance.

## A. Residual personalization

Use `target = source_ppm + Delta_target(83D)` with a single gas-specific Ridge
residual head. It adds one target Ridge map per gas and preserves the source
sufficient-statistics property. It must beat canonical M83 and canonical M84 on
the same grouped-bootstrap C3/C4/C5 endpoints. Stop if pooled Delta-RMSE CI is
not below zero or any target materially reverses.

## B. FedRidge reliability prior

Add source ppm and a source-only uncertainty statistic derived from federated
validation residuals; do not use target test. This preserves one-shot aggregate
exchange if uncertainty is represented by aggregate SSE/count. It must beat the
canonical one-prior M84 and confidence-only QC at identical coverage. Stop if
the grouped-bootstrap benefit or QC AURC advantage is absent.
