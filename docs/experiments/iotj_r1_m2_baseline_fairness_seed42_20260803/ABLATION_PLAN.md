# R1-M2 Baseline Fairness Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| H-R1M2-01 | Training-domain access | target calibration only; GAPS sources + same calibration | TCN, seed42, C5 calibration/test, declared update budget | C5 macro-F1 | Quantifies whether cross-device source knowledge helps | Equal target-label count; no C5-test selection | Stop on any test-driven tuning |
| H-R1M2-02 | Optimization topology | centralized pooled source; FedAvg source | TCN, source samples, 25x5 epochs, Adam 5e-4, seed42 | C5 macro-F1 | Quantifies federation-only effect | Same sample exposure and final-round evaluation | Stop if source split or epoch budget differs |
| H-R1M2-03 | Federated optimizer | FedAvg; FedProx mu=0.01 | C1/C2, no target labels, TCN, 25x5, seed42 | C5 macro-F1 | Tests whether standard heterogeneity regularization explains the gain | One predeclared mu; no tuning | Stop if target data enter training |
| H-R1M2-04 | GAPS-specific mechanisms | FedAvg + matched server adapter; full GAPS B5 | Same target calibration, adapter steps/losses, backbone, seed42 | C5 macro-F1 | Isolates client alignment/replay/decoupling/selective aggregation | Verify CE-only client loss and sample-weighted FedAvg in logs | Stop if adapter settings diverge from B5 |
| H-R1M2-05 | Calibration-transfer method | regularized DS + frozen FedAvg; GAPS B5 | C5 calibration/test, seed42, classifier checkpoint lock | C5 macro-F1 | Benchmarks a domain-specific e-nose transfer route | DS hyperparameter selected only inside calibration | Stop if matched strata fail or test labels are used |

## Required baselines

- Existing canonical GAPS B5 seed42: read-only reference.
- Existing source-only FedAvg/A0 seed42: hash-pinned read-only reference, with old-code provenance disclosed.
- Five new registered runs in `EXPERIMENT_MATRIX.csv`.

## Resource budget and execution order

1. Validate data/checkpoint hashes and DS stratum matching without reading test outcomes.
2. Run target-only and centralized source-only locally.
3. Run FedProx source-only distributed over C1/C2.
4. Run FedAvg + same target adapter distributed over C1/C2 with C5 calibration on the server.
5. Lock DS regularization on C5 calibration, run the single sealed C5 test evaluation, then evaluate all completed checkpoints through one common evaluator.

Formal distributed runs are sequential. Each configuration has one seed42 run; crashes may be restarted only with an identical manifest.

## Unknown or conflicting protocol fields

- Historical A0 comes from a prior code revision. Its checkpoint identity is valid, but source-tree equivalence is not assumed.
- Peak-memory comparability depends on host telemetry availability and will be marked unavailable rather than imputed.
- Statistical uncertainty cannot be inferred from one seed. The paper must label these comparisons seed42-only.
