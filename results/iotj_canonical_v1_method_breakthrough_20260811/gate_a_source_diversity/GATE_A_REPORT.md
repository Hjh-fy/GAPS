# Gate A Source-diversity / Federated-DG Report

## [Scientific Question]

Does adding C3/C4 source domains improve C5 zero-shot Macro-F1, and does exact GAPS-DG-P add at least one percentage point beyond matched FedAvg?

## [Protocol]

S2 endpoints are immutable reused round25 results. S4 uses C1-C4, 25 rounds, LE1, batch32, Adam 5e-4, seed42. C5 was absent from every training API and command. C3/C4 use the pre-frozen derived canonical role view; C1/C2/C5 are byte-identical to canonical-v1.

## [Primary Result]

- S2 FedAvg C5 Macro-F1: 0.368610
- S4 FedAvg C5 Macro-F1: 0.386605
- S2 GAPS-DG-P C5 Macro-F1: 0.316017
- S4 GAPS-DG-P C5 Macro-F1: 0.461595

## [Negative Result / Limitation]

This is seed42 and changes both source-domain count and labeled source-data composition. It is a C5 hardest-target sensitivity, not a pure causal domain-count ablation or universal DG result.

## [Leakage Audit]

Both S4 completion markers were locked before C5 test evaluation. The S4 training protocol contains no C5 path, X, Y, phase, concentration, statistics, or calibration access. Target test was not used for tuning, stopping, or checkpoint selection.

## [Decision]

- `SOURCE_DIVERSITY_SUPPORTED`
- `SOURCE_DG_PROMISING`

## [Paper Implication]

Only the registered C5 source-diversity sensitivity wording is permitted. Prototype-DG superiority requires the pre-registered one-point matched gain and source-retention gate.

## [Next Action]

`CREATE_MULTI_SEED_PROPOSAL_ONLY`. Gate B still uses the frozen S2 source endpoint; no source-count substitution or new run is made there.
