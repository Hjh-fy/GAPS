# GAPS Method-Breakthrough Validation Design

## Objective

Run three strictly ordered, fail-closed gates without changing canonical-v1 preprocessing, historical A4/A0T/GAPS-SSDA endpoints, C5 target identities, R84, Ridge alphas, or QC:

1. Gate A tests whether C5 zero-shot behavior changes when source diversity expands from C1+C2 (S2) to C1+C2+C3+C4 (S4), and whether the already frozen GAPS-DG-P mechanism adds value at either source count.
2. Gate B tests whether post-hoc C5 commissioning can be localized to a small target-specific artifact while the source global remains immutable.
3. Gate C audits whether C5 classification errors have heterogeneous downstream regression cost and whether the historical A0T-versus-A4 S_ALL RMSE gap is supported beyond one abnormal sample or file.

## Selected design

### Gate A role view

The canonical-v1 directory is immutable and has source train arrays only for C1/C2. S4 therefore uses a new derived role-view dataset. C1/C2 and C5 are copied byte-for-byte. Only C3/C4 are re-roled: their complete canonical calibration+test pools are partitioned per client and per class×concentration stratum into 70% train, 10% calibration, and 20% test with fixed seed42 client-local RNG. The derived view writes to a new directory. C5 calibration/test physical identities and file hashes must equal canonical-v1. The canonical-v1 directory is never edited.

The legacy C1-C4-source metadata was inspected but cannot serve as a window-identity map because it omits physical window start/end fields. Inferring those identities would create unverifiable provenance. The explicit new C3/C4 partition is therefore frozen before execution and is reported as a source-diversity-plus-added-source-data sensitivity, not an optimizer-controlled single-factor causal comparison.

Rejected alternatives:

- Treating only C3/C4 calibration windows as source train would make client data budgets and split semantics incomparable.
- Reusing the legacy preprocessed feature arrays would violate the canonical-v1 preprocessing constraint.
- Using C3/C4 target-test arrays for source training in place would destroy the requested per-source held-out evaluation.

S2 FedAvg and S2 GAPS-DG-P are reused only after manifest, round, LE, seed, optimizer, dataset, checkpoint, and target-access audits pass. S4 uses 25 rounds, LE1, batch 32, seed42, Adam 5e-4, fixed round25, with C5 completely absent from training APIs and commands. S4 FedAvg and S4 GAPS-DG-P differ only by the exact G2 prototype-alignment mechanism (`lambda_proto=0.05`).

### Gate B selective reuse and lightweight artifacts

All methods start independently from source state fingerprint `cad6726ec29fb574314a5f2a45ed9800d1d90906b81cbd3ba8f8efb48a0df5d7` and use the same 320 C5 calibration identities for 100 Adam steps at 5e-4, batch 32, seed42.

- B0 reuses G1 Source-only.
- B1 reuses G1 Posthoc A0T-full.
- B2 trains only `classifier`.
- B3 reuses G1 `target_head`, whose manifest proves it trains `feat_proj + classifier` and nothing else.
- B4 trains a rank-4 residual adapter after the normalized 64-D feature plus the classifier. The trained linear adapter is folded exactly into the classifier weight (`W_fold = W(I + BA)`) so the endpoint remains a standard checkpoint and does not change runtime architecture. If an exact fold/serialization test fails, B4 is `NOT_IMPLEMENTED` and no model refactor is allowed.

C1/C2/pooled retention is diagnostic only because personalized checkpoints are never shared back to the immutable source global.

### Gate C cost audit

Gate C performs no classifier or regressor training and no target-test selection. It reconstructs C5 calibration rows using the frozen A0T/A4 classifier checkpoints, frozen FedRidge H1, frozen per-class R84 Ridge models and fixed alphas. For every calibration row it forces all four routes and records raw-ppm excess squared error relative to the correct route. Negative off-diagonal mean costs are clipped to zero only in the published primary policy matrix; raw values and count/median/P90 remain preserved.

The C5 test is used only after the calibration cost matrix is locked. Existing per-window A0T/A4 S_ALL and Oracle_ALL streams provide actual misroute identities and paired error decomposition. Uncertainty uses at least 2,000 seed42 grouped bootstrap replicates over raw filename/experiment groups. No bootstrap result changes a model, threshold, checkpoint, or matrix.

## Leakage and stopping controls

- C5 X/Y/statistics/calibration are unavailable during all Gate-A source training.
- C5 test is unavailable for training, hyperparameters, stopping, checkpoint choice, B-method selection, and cost-matrix construction.
- Each gate writes a protocol lock before execution and a sealed-test-open marker only after all required endpoints are locked.
- Negative Gate A does not trigger lambda or source-composition search.
- Negative Gate B does not trigger architecture/rank/LR/step search.
- Gate C ends this execution. Gate D/E/F are never started automatically.

## Output roots

- Plan/audit/registry: `docs/experiments/iotj_canonical_v1_final/method_breakthrough_20260811/`
- Results: `results/iotj_canonical_v1_method_breakthrough_20260811/`
