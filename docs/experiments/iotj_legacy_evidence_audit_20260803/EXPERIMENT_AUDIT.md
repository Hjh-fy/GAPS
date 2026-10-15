# IoT-J Legacy Experiment Audit

## Verdict

No requested historical row qualifies as `main_paper_ready`. `9` classification/component rows are `supplement_only`; all `6` requested cross-direction regression rows are `historical_diagnostic_only` because their metric-producing assets are not direction-bound.

## Blocking findings

1. **Cross-direction regression replay unavailable.** F4/F5/R1–R4 have classification checkpoints and summaries, but no verified row-level regression truth/prediction pair with an explicit route schema. Recomputing NRMSE would require guessing an unrelated pipeline identity, so replay was not run.
2. **Seed and commit are unbound for the legacy F1–F5/R1–R4 matrix.** These rows cannot be called seed42 and cannot support across-seed stability.

## Major findings

1. **Current protocol compatibility is limited.** Only historical F2 shares the C1/C2→C5 role labels and named 320/1360 window split. It still differs from the final canonical ECS-C2 topology, code identity, and corrected/frozen method evidence. F1/F3/F4 change source sets; F5 and R1–R4 change target scope or direction.
2. **Historical semantics.** The old fixed-da-strong matrix is not final B5. B1–B5 are corrected mechanism screens, but are single-seed, older-topology and test-visible; the historical B5 name must not be equated with the final five-seed B5 evidence.
3. **F4 provenance conflict.** The 2026-06-30 F4 summary and 2026-07-08 canonical recovery report different target metrics. The inventory follows the documented recovery for the active row and preserves the conflict in `notes`/`provenance_status`.
4. **Target-test boundary.** Historical matrix and B1–B5 results were visible during development. No per-run fitting to test is documented, but absence of test-driven method screening is not established. They cannot support untouched-test or prospective-confirmatory wording.

## Routing audit

- Classification rows have no regression route assumption.
- Requested cross-direction regression rows lack persisted prediction columns and route identity; classification-vs-regression routing consistency is therefore `not_auditable`.
- Existing formal C1/C2→C5 oracle-route records are a different experiment family and were intentionally excluded.

## Evidence-tier decision

- `main_paper_ready`: none.
- `supplement_only`: F1–F4 classification context and B1–B5 historical component screening.
- `historical_diagnostic_only`: F5/R1–R4 classification context and all requested cross-direction regression placeholders with unknown metrics.

## Integrity boundary

The audit reads historical summaries, run configs, confusion matrices and checkpoint bytes only for identity hashing. It does not load checkpoints, execute inference, train models, open formal C5 test arrays, or modify frozen assets.
