# Experiment Plan: canonical-v1 Method Breakthrough

## Research brief and scope

- Brief source: user-frozen method-breakthrough specification dated 2026-08-11.
- Target venue/audience: IEEE IoT-J submission evidence.
- Resource budget: seed42 only; Gate A two new 25-round S4 Flower runs; Gate B at most two new 100-step post-hoc runs; Gate C analysis only; no Gate D/E/F.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-A1 | Adding C3/C4 as canonical-preprocessed source clients improves C5 zero-shot generalization. | S2 FedAvg | S4 FedAvg | C5 Macro-F1 | Fixed-endpoint S4-minus-S2 comparison | Meaningful positive C5 F1 change reported with source pooled retention; no universal claim from seed42. |
| H-A2 | Exact frozen GAPS-DG-P adds value beyond FedAvg at a fixed source count. | FedAvg at S2/S4 | GAPS-DG-P at same source count | C5 Macro-F1 | Matched source-count algorithm comparison | At least +0.01 C5 Macro-F1 with no meaningful pooled-source degradation. |
| H-B1 | Localized post-hoc personalization can match full A0T with substantially fewer trainable parameters. | Posthoc A0T-full | classifier-only, projection+head, rank-4 adapter+head | C5 Macro-F1 | Same checkpoint/identities/budget comparison | Within 0.005 Macro-F1 of A0T-full and materially fewer trainable parameters; choose simplest sufficient method. |
| H-C1 | C5 route errors have heterogeneous downstream regression cost. | Correct-route R84 prediction | Forced wrong routes | Excess squared error in raw ppm | Calibration-only 4x4 cost matrix | Heterogeneous costs supported across multiple samples/files, not one abnormal file. |
| H-C2 | The A4-versus-A0T C5 S_ALL RMSE difference reflects different costly misroutes rather than correct-route mapping. | A0T frozen stream | A4 frozen stream | Paired RMSE and excess-SE differences | Test-only diagnostic plus grouped bootstrap | CI/decomposition and file support determine motivation; no classifier claim is tuned from test. |

## Fixed protocol

- Source clients: Gate A S2=`C1;C2`, S4=`C1;C2;C3;C4`; Gate B source checkpoint trained on `C1;C2`.
- Target clients: `C5` only.
- Split protocol: derived S4 role view uses frozen physical identities from the repository's C1-C4-source/C5-target map; C5 calibration/test identities must exactly equal canonical-v1.
- Model/checkpoint policy: canonical backbone; fixed round25; every post-hoc endpoint independently reloads source state fingerprint `cad6726e...d5d7`.
- Seeds: 42 only.
- DA/calibration/QC controls: GAPS-DG-P exact G2 `lambda_proto=0.05`; C5 calibration N=320; 100 Adam steps at 5e-4; QC and R84 unchanged; Gate C policy costs are calibration-only raw ppm.

## Risks, unknowns, conflicts, and stopping rules

- S4 is a source-composition sensitivity study, not strict experiment-independent generalization evidence.
- C3/C4 role changes are explicit and use a new derived dataset; the canonical-v1 directory is read-only.
- Gate A negative results stop DG expansion; no lambda/source search.
- Gate B B4 becomes `NOT_IMPLEMENTED` if exact fold/serialization is not clean; no refactor or rank search.
- Gate C negative result stops cost-aware routing. A positive result records `GO_GATE_D` only; Gate D is not executed in this task.

