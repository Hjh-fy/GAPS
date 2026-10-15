# Controlled-factor and stopping audit

| Phase | Factor | Levels | Held constants | Expected evidence | Stop condition |
|---|---|---|---|---|---|
| P1 | source algorithm | FedAvg / GAPS-DG-P | S4 clients, canonical role view, backbone, Adam 5e-4, 25 rounds, LE1, batch32, seed pairing | paired C5 zero-shot F1 stability | exactly seeds 41/42/43 |
| P2 | source initialization | I0 / I1 / I2 | Full A0T, fixed 100 steps, same budget identities, sealed C5 test | zero-shot vs commissioning and label-efficiency decomposition | exactly B20/B05 |
| P3 | method identity | conditional registered source method | B20, Full A0T, canonical H1, frozen C5 alphas | one immutable argmax baseline | one selected identity only |
| P4 | routing rule | argmax / expected cost | same probabilities, R84, cost matrix, test, no tunable parameter | classification-regression tradeoff and grouped CI | apply frozen guardrails; no safe-top2 implementation |

Confounds that must remain explicit: S2 versus S4 changes both source-domain count and source sample composition; P1 has only three seeds; P4 cost estimates come from one target calibration pool.

