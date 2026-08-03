# IoT-J P0 Routing Simplification Study

## Research brief and scope

- Goal: determine the simplest reliable classification router using one common CE-only, LE=1, pure-FedAvg source trajectory.
- Budget: seed42 only; 25 global rounds; C1/C2 source clients; C5 target; no low-calibration, regression, QC, Pi benchmark, cross-direction, or grouped server ablations.
- Formal comparison: round 25 only. Round-wise C5 curves are retrospective diagnostics and cannot select a checkpoint or configuration.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| P0-H1 | CE-only LE=1 FedAvg learns a usable source router. | round-1 source-only | round-25 source-only | C5 Macro-F1 | 25-point post-hoc curve | descriptive round-25 performance; no predeclared favorable threshold |
| P0-H2 | Simple target CE is at least as effective as full DA at fixed round 25. | full target adapter | 100-step target CE | C5 Macro-F1 | fixed round-25 comparison | compare signed difference without significance language |
| P0-H3 | One-time commissioning after source FL is sufficient relative to per-round adaptation. | round-wise adapter curves | round-25 commissioning | C5 Macro-F1 | convergence shape plus formal round-25 row | interpretation remains descriptive, seed42 only |

## Fixed protocol

- Dataset: `dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid`.
- Source: C1/C2, 2,360 training windows each. Target: C5, 320 calibration and 1,360 sealed test windows.
- TCN classification backbone unchanged; Adam; LR 5e-4; batch size 32; seed42.
- P0-A: 25 rounds, one local epoch, CE-only, sample-weighted FedAvg, no target access.
- P0-B1/B2: each round independently reloads its original source checkpoint; 100 commissioning steps; no adapted inheritance.
- C5 test: evaluation only after P0-A and the applicable commissioning operation complete.

## Risks and stopping rules

- Any contract, hash, split-role, checkpoint-count, or topology preflight failure stops execution fail-closed.
- Single seed does not support variance, confidence interval, p-value, or stability claims.
- P0 is a new LE=1 routing protocol and is not a single-factor ablation against historical five-local-epoch B5.
