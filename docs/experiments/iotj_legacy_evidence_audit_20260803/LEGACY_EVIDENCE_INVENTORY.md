# IoT-J Legacy Evidence Inventory

- Schema: `gaps.iotj.legacy_evidence_inventory.v1`
- Audit mode: read-only; no training, checkpoint inference, model fitting, or benchmark.
- Scope: F1–F5/R1–R4 historical classification matrix, B1–B5 corrected server-adaptation screen, and requested F4/F5/R1–R4 regression provenance.
- Evidence tiers: `main_paper_ready=0`, `supplement_only=9`, `historical_diagnostic_only=11`.

## Inventory boundary

The complete legacy matrix assets are local historical files outside the current worktree's tracked result set. The CSV manifests preserve that storage boundary. An existing path is not treated as a portable or paper-ready artifact unless seed, commit, checkpoint, split, and metric provenance are bound.

## Classification matrix

- F1–F3 use the 2026-06-30 final target summary.
- F4/F5/R1–R4 use the 2026-07-08 canonical full-name recovery recorded in the experiment notebook.
- F4 has two traceable result sources with different values. The inventory uses the notebook-designated recovery and records the earlier value as a conflict; it does not average or silently replace either source.
- Clean-matrix Macro-F1 is recomputed only from persisted confusion matrices. Accuracy, NLL, and ECE remain copied reported values.

## B1–B5 server adaptation

B1 is CORAL, B2 is conventional global/class MMD², B3 is cross-domain same-class/same-phase stage MMD², B4 is corrected Wasserstein-min adversarial alignment, and B5 is their predeclared combination on the shared semantic core. All five are seed-42 historical screens using the older Windows-PC C2 topology and test-visible development context. They are supplementary mechanism evidence, not final-B5 five-seed causal evidence.

## Cross-direction regression

No direction-bound frozen row-level regression prediction stream was found for F4/F5/R1–R4 in the canonical matrix roots or the tracked evidence indexes. Therefore `NRMSE_CC` and `NRMSE_ALL` remain `unknown`, and evaluation replay is explicitly `NOT_RUN_NO_FROZEN_PREDICTION_ASSET`. C1/C2→C5 formal R4/oracle metrics and the broad C4/C5→C1/C2/C3 legacy pipeline were not substituted because they use different source/target and routing identities.

## Machine-readable files

- `classification_cross_direction_summary.csv`
- `server_adaptation_component_summary.csv`
- `regression_cross_direction_summary.csv`
- `EXPERIMENT_AUDIT.md`
- `sha256_index.json`
