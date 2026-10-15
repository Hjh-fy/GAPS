# Gate 2 Experiment Audit

## Audit scope

Audit the fixed FedAvg versus GAPS-DG-P source-only comparison for unseen C5 zero-shot classification.

## Findings

| Finding ID | Severity | Check | Evidence | Impact | Status |
|---|---|---|---|---|---|
| G2-F01 | informational | No target training access | server/client commands contain C1/C2 paths only; protocol records target X/Y/phase false | valid source-only DG test | closed |
| G2-F02 | informational | Fixed source protocol | 25 rounds, LE1, batch32, Adam 5e-4, seed42 | matches canonical baseline scale | closed |
| G2-F03 | informational | Intended loss | round1 inactive; rounds2-25 receive 12 prototypes and alignment is active | mechanism executed | closed |
| G2-F04 | informational | Extra mechanisms | selective aggregation, replay, prototype-MMD diagnostic, regression, and server DA are disabled/inactive | isolates prototype alignment | closed |
| G2-F05 | informational | Phase safety | source phase comes from acquisition window metadata | no target-dependent phase | closed |
| G2-F06 | major | Seed scope | seed42 only | descriptive endpoint evidence, not population uncertainty | closed with limitation |
| G2-F07 | informational | Preflight failures | three lock-only directories were preserved; each contains no process log or round artifact | no duplicate or partial training endpoint | closed |

## Leakage assessment

C5 was absent from source commands, loaders, prototype aggregation, and checkpoint selection. C5 test was opened only after the GAPS-DG-P round25 completion marker and checkpoint hash were verified. No target metric selected lambda, warm-up, or endpoint.

## Verdict: approved negative result

The comparison is complete and supports `SOURCE_DG_NOT_SUPPORTED`. It does not support further result-driven prototype tuning. The negative result must be preserved in the method redesign decision.

