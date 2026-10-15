# Gate B Lightweight Post-hoc Target Personalization

## [Scientific Question]

Can a new C5 node reach full A0T performance by personalizing only a small endpoint rather than the complete source model?

## [Protocol]

All methods use the immutable canonical S2 round25 source checkpoint and the same 320-window canonical-v1 C5 calibration set. B2 and B4 independently reload the source, use 100 Adam steps at 5e-4 with seed42, and lock step100 before C5 test evaluation. B1 and B3 are immutable audited G1 endpoints. The rank-4 adapter is exactly folded into the ordinary classifier for deployment.

## [Primary Result]

- Full A0T C5 Macro-F1: 0.976544
- Classifier-only C5 Macro-F1: 0.368610
- Projection+Head C5 Macro-F1: 0.368610
- Rank-4 adapter C5 Macro-F1: 0.415596

## [Negative Result / Limitation]

This is seed42 on C5 with a fixed 100-step budget. Source-retention scores describe the hypothetical adapted checkpoint; the operational global source checkpoint remains immutable.

## [Leakage Audit]

Only the canonical C5 calibration loader entered B2/B4 adaptation. The C5 test manifest was opened after both new endpoints were locked. No target-test checkpoint selection or hyperparameter search occurred.

## [Decision]

`FULL_ADAPTATION_REQUIRED`; selected path: `a0t_full`.

## [Paper Implication]

The method story may claim lightweight commissioning only if the registered 0.5-point and parameter-reduction gates are met; otherwise full target adaptation remains necessary.

## [Next Action]

Proceed to the already frozen read-only Gate C routing-cost audit. Do not start Gate D/E/F.
