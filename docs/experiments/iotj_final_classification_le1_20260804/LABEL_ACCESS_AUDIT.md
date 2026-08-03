# Target Information and Label-Access Audit

## Static audit contract

| Method | Calibration API fields | Test API fields before completion | Conditional target losses | Target CE | Selection use |
|---|---|---|---|---|---|
| E0 | x | unavailable | unavailable | unavailable | none |
| FedAvg/FedProx/SCAFFOLD | none | unavailable | unavailable | unavailable | none |
| CORAL/MMD/DANN | x-only tensor loader | unavailable | unavailable | unavailable | fixed step 100 only |
| A0-A3 | none | unavailable | unavailable | unavailable | round 25 only |
| GAPS/A4-A6 | x, class, phase | unavailable | registered method terms only | configured weight 0 | round 25 only |

The E2 adaptation function accepts `target_x_loader` and has no target class/phase/concentration argument. Its target dataset returns one tensor rather than a tuple. GAPS access is separately authorized because its formal method definition includes class/phase-conditioned server alignment; this does not authorize concentration or target CE.

## Runtime ledger

Every target access writes method, stage, split, requested fields, purpose, allow/deny, severity and reason to a JSONL ledger. Any target-test request outside `final_evaluation` raises `TargetTestLeakageError` with `HARD_FAIL`. The final-evaluation unlock requires the exact fixed-endpoint completion marker and yields a one-use method-target token.

## Pre-run status

Static API and policy tests are required to pass before the formal lock. Runtime rows and final PASS/HARD-FAIL counts will be appended after execution. No unavailable target field is represented by a zero-weight tensor passed into adaptation; it is absent at the loader/API boundary.
