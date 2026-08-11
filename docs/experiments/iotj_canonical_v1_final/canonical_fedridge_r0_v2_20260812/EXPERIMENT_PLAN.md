# Experiment Plan

## Research brief and scope

- Brief source: approved R0-v2 design at
  `b41fee1d5bd64a19d6fefcad5fde610183856202` and this frozen protocol.
- Target venue/audience: IEEE Internet of Things Journal manuscript evidence
  boundary; no manuscript-body edit is part of this plan.
- Resource budget: `unknown`; formal execution remains blocked, so no runtime,
  accelerator, memory, or wall-time assumption is made.
- Study status: `DESIGN_FREEZE_READY_FORMAL_NOT_STARTED`;
  `formal_execution_started=false`.
- Prior decisions remain C0=`V1_INTERLEAVED_RETAINED` and original
  R0=`R0_EXACT_RECOVERY_NOT_ESTABLISHED`.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-R0V2-NUM | Formal execution will determine whether every preregistered gate passes; either registered decision is admissible, with no expected direction. | POOLED-RIDGE-SAME-ROWS: pooled Ridge using the same scaler semantics, row order, alpha, solver, and intercept policy | Source-only sufficient-statistics reconstruction with float64 mergeable central moments in C1-then-C2 order | Registered per-gas hard-gate conjunction and one registered R0-v2 decision | Four per-gas gate records and one registered PASS/FAIL decision; no direction or numeric outcome is predicted | `FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED` only if every hard Boolean is true for exactly gases 0,1,2,3; otherwise `R0_V2_FAILED` |

## Fixed protocol

- Source clients: ordered `C1;C2`; no target clients.
- Split protocol: per client train=2360 for candidate fit,
  calibration=320 for distributed source SSE/count alpha selection,
  train+calibration=2680 total/670 per gas for final refit, and test=680
  total/170 per gas opened only after source alpha/model locks.
- Model/checkpoint policy: four per-gas 104D `CanonicalRidgeModelV2` models;
  no pre-run checkpoint; future immutable `source_alpha_lock.json` and
  `model_lock.json` only under the registered result path.
- Seeds: `42` with note `deterministic numeric reconstruction; seed unused`.
- DA / calibration / QC controls: `none / none / none`. “Calibration” here is
  the model-control mode; the named source calibration split retains its
  frozen alpha-selection role.
- Dataset/config/code: `dataset/iotj_canonical_v1` at aggregate SHA
  `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`;
  `protocol_manifest.json`; task base
  `6668dc5db83428a2d957d962d6a5fa4bb5dc2430`.
- Held constants: 5 Hz, 50x8, 83D/104D features, float64, `M2/n`, strict
  `raw_scale < 1e-9`, C1-then-C2 merge, registered alpha grid and first tie,
  `numpy.linalg.pinv`, unregularized intercept, exact Task 4 gates.
- Ablations: none. This plan contains one paired reconstruction comparison and
  one executable configuration only.

## Risks, unknowns, conflicts, and stopping rules

- Unknown: execution resource budget.
- Unknown until authorized execution: Python/NumPy/platform/BLAS/LAPACK
  environment metadata. It may be recorded but cannot change a threshold.
- Conflicts: none.
- Stop before the affected row on any provenance, hash, count, role, order,
  feature, lock, access, finite-value, or gate mismatch. Preserve the evidence
  and do not infer support.
- Do not adjust tolerances, model, solver, alpha, features, or aggregation after
  observing evidence.

## Planner-to-registry handoff

- From: `experiment-planner`; to: `experiment-registry`.
- Inputs: approved design, Task 4 code at the task base, this plan, and
  `protocol_manifest.json`.
- Completed: unique ID, exact canonical fields, source role/order, empty target
  role, one-row matrix, and pre-run status checks.
- Blocking Evidence Gap: formal source-only execution has not occurred.
- Requested action: register the one frozen candidate with evidence status
  `blocked_pending_execution` and await a separately named freeze commit.
- Read-only prerequisites: approved design, v2 numerical module, and canonical
  dataset.

Decision vocabulary retained for the handoff:
`FEDRIDGE_ALGEBRAIC_EXACT_NUMERICAL_EQUIVALENCE_ESTABLISHED` or
`R0_V2_FAILED` only.
