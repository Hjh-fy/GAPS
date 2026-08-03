# Pre-execution Experiment Audit

## Scope

This audit covers readiness of the frozen seed42 classification comparison: standard FL baselines, canonical x-only UDA references, Full GAPS across C3/C4/C5, and the C5 A0-A6 hierarchy. It authorizes execution only and is not result evidence.

## Findings

| Finding ID | Severity | Check | Evidence | Status |
|---|---|---|---|---|
| PRE-01 | blocking | P0A immutable import | round25 imported read-only; ordered state-content fingerprint `c0379777c63c7a5b0910e5d2ab8a1c37869692db14fd940e0a86778c9f5a1769`; whole-file SHA retained only for provenance | PASS |
| PRE-02 | informational | historical FedProx comparability | historical LE5 excluded; new LE1 run registered with mu0.01 | PASS |
| PRE-03 | informational | historical GAPS comparability | historical warmup3 excluded; new warmup5 runs registered | PASS |
| PRE-04 | major | optimizer disclosure | `PROTOCOL.md` contains optimizer, optimizer_lr and optimizer_note; canonical SCAFFOLD SGD is explicitly not optimizer-controlled against Adam methods | PASS with disclosed limitation |
| PRE-05 | blocking | E2 target isolation | target loader returns x tensor only; adaptation API has no target class/phase/concentration argument; all nine specs use fixed step100 | PASS |
| PRE-06 | blocking | target-test seal | hard-fail policy and one-use fixed-endpoint unlock implemented and tested; generated training/adaptation commands contain no target-test path | PASS pre-run |
| PRE-07 | major | seed coverage | seed42 only by explicit decision; protocol prohibits stability/generalization claims from seed variation | PASS with scope limitation |
| PRE-08 | blocking | SCAFFOLD numerical validity | C1/C2-only gate passed loss decrease, source discrimination, finite, gradient norm and parameter norm checks; no LR search and no target information | PASS |
| PRE-09 | blocking | method-specific target fields | calibration fields are separately registered for E0/E1/E2/E3/E4; target-test leakage is absolute HARD FAIL | PASS |
| PRE-10 | blocking | ablation loss activity | exact requested schema is wired into client and server paths; A4/A5 actual-input activity tests pass | PASS pre-run |
| PRE-11 | blocking | selective warm-up boundary | one helper drives code/logs; tests prove rounds1-5 complete FedAvg and round6+ selective; formal runs fail closed if semantic inputs are absent after warm-up | PASS |

## Verification evidence

- Focused protocol suite: 57 passed.
- Relevant Flower/DA/baseline regression suite: 89 passed.
- `compileall`: passed.
- Immutable checkpoint import: passed.
- C1/C2-only SCAFFOLD numerical gate: passed all seven checks.
- Strict pre-run audit: 11/11 checks passed with zero blocking failures.

## Leakage and reproducibility assessment

E2 consumes target calibration x only. Full GAPS and A4-A6 consume target calibration x/class/phase and not concentration; target CE is fixed at zero. A0-A3 and E1 consume no target calibration in training. Target test remains sealed until the exact method-target fixed endpoint has an immutable completion marker, after which only a one-use final-evaluation token can expose x/class.

Pre-run reproducibility inputs are complete: fixed matrix and protocol hash, ordered checkpoint provenance, generated three-host commands, source numerical gate, target policy, endpoint locks and strict audit. Training logs, round checkpoints, runtime access ledgers, metrics, predictions, cost tables and SHA index remain post-run requirements.

## Verdict: PASS for formal execution

Formal execution may start from the pre-run freeze commit. After `formal_training_started.lock` is written, no tuning, matrix mutation, target-test selection, learning-rate search, new seed or automatic method optimization is permitted.
