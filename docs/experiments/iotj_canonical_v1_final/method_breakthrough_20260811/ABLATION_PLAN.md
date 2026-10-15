# Ablation Plan

| Hypothesis ID | Factor | Levels | Held constants | Primary metric | Expected Evidence | Confound check | Stopping rule |
|---|---|---|---|---|---|---|---|
| H-A1 | Source count | S2; S4 | preprocessing, backbone, optimizer, rounds, LE, seed, C5 endpoint | C5 Macro-F1 | source-diversity sensitivity | role-view and per-source sample counts explicit | no extra source sets |
| H-A2 | Source-DG mechanism | FedAvg; exact GAPS-DG-P | fixed source set | C5 Macro-F1 | algorithm effect at S2/S4 | source pooled F1 and representation metrics | no lambda search |
| H-B1 | Trainable scope | full; classifier; projection+head; rank4+head | source fingerprint, C5 identities, optimizer, LR, steps, seed | C5 Macro-F1 | simplest sufficient personalization | C1/C2 retention and parameter count | no rank/LR/step search |
| H-C1/H-C2 | Forced route / actual router | four forced routes; A0T; A4 | frozen H1, R84, alpha, C5 split | raw-ppm excess SE and S_ALL RMSE | heterogeneous route cost and paired decomposition | grouped filename bootstrap | stop before Gate D |

## Required baselines

S2 FedAvg, S2 GAPS-DG-P, B0 Source-only, B1 A0T-full, and the frozen A0T/A4 R84 streams are mandatory. Reuse requires immutable hash and protocol equality.

## Resource budget and execution order

Gate A then Gate B then Gate C. Two new S4 25-round runs, two new 100-step B runs, and analysis-only Gate C. Seed42 only.

## Unknown or conflicting protocol fields

No unresolved conflict is permitted at execution. The absence of C3/C4 source train arrays in canonical-v1 is resolved by a new derived role view with exact physical-identity and C5 hash gates.

