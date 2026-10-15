# Planning Handoff

- from_skill: `experiment-planner`
- to_skill: `experiment-registry`, followed by implementation planning
- input_artifacts: `EXPERIMENT_PLAN.md`, `EXPERIMENT_MATRIX.csv`, `ABLATION_PLAN.md`, `PRE_EXECUTION_AUDIT.md`
- version: design commit to be recorded after commit
- completed_checks: protocol boundary, baseline coverage, seed, optimizer disclosure, checkpoint provenance, E2 label-access policy, historical reuse conflicts
- unresolved_unknowns: none
- unresolved_blockers: immutable P0A import audit; E2 label-access implementation tests; sealed-test runtime gate
- blocking_evidence_gap: no new run results exist yet; all Evidence remains draft
- requested_next_action: user spec review, then implementation plan and TDD execution
- read_only_assets: all existing `results/`; P0A source checkpoint and manifests; existing datasets
