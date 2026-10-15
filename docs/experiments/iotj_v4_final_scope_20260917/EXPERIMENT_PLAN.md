# Experiment Plan

## Research brief and scope

- Brief source: user-approved IoT-J v4 scope discussion dated 2026-09-17, constrained by the 2026-09-14 final-experiment brief.
- Target venue/audience: IEEE Internet of Things Journal reviewers and readers.
- Resource budget: no new FL/DA training; reuse the 20% R83/R84 pair; exactly six new C5 Ridge fits at 15%/10%/5%; QC is post-processing only; seed42 and frozen canonical identities.
- Design source: `docs/superpowers/specs/2026-09-17-iotj-v4-final-evidence-scope-design.md`.

## Hypotheses

| ID | Falsifiable hypothesis | Baseline | Intervention | Primary metric | Expected Evidence | Acceptance criterion |
|---|---|---|---|---|---|---|
| H-CLS-01 | Canonical FL and DA endpoints can be compared descriptively within explicit information regimes without inserting M0 or legacy probes into the main ranking. | FedAvg within source-only FL | FedProx, SCAFFOLD, CORAL, MMD, DANN, A0T, GAPS/A4 in declared regimes | Accuracy; Macro-F1 on C3/C4/C5/pooled | One hash-traceable grouped table with target-information fields | Every row reproduces an authoritative canonical endpoint and no cross-regime equal-information claim is made |
| H-R84-01 | At each frozen C5 budget, the incremental effect of the one-dimensional frozen source prior is measurable as paired R84-minus-R83 error; its sign is not assumed. | R83 target-only | R84 = same 83D inputs plus frozen R0-v2 source prediction | delta NRMSE_range; delta RMSE ppm | Four paired budget effects with conditional grouped-bootstrap CIs | Complete, identity-matched R83/R84 pairs at 20/15/10/5 with no test-driven selection |
| H-R84-02 | Source-prior effects may differ by gas and are not assumed to be uniform or monotonic as budget decreases. | Per-gas R83 | Per-gas R84 | Per-gas delta NRMSE and delta RMSE | Four-gas effect table at each budget | All gases and all budgets reported, including zero or adverse effects |
| H-QC-01 | The frozen confidence risk score yields an observable NRMSE--coverage tradeoff over 70%--100%; monotonicity is not assumed. | 100% no-QC endpoint | Retain samples by the frozen confidence ranking | Actual coverage; accepted NRMSE_range; AURC | Pooled primary curve and target-specific secondary curves | Values reproduce authoritative Q1 rows and HC90/HC95 remain calibration-locked markers |
| H-QC-02 | If canonical equal-retention random post-processing passes identity audit, it quantifies whether confidence ranking adds information beyond rejecting the same number of samples. | Same-count random retention | Frozen confidence ranking | NRMSE difference and random interval | Optional canonical random-reference band | Same prediction population and retained counts; otherwise mark `SKIPPED_IDENTITY_NOT_CONFIRMED` |

## Fixed protocol

- Source clients: classification and R0-v2 source prior use the exact canonical client roles and ordered source locks; R83 itself is target-only.
- Target clients: classification C3/C4/C5; budget experiment C5 only; QC C3/C4/C5 with pooled micro concatenation.
- Split protocol: canonical-v1 frozen calibration/test identities; C5 budgets are nested 80/160/240/320 rows over 40 class-by-concentration strata with zero exact test identity overlap.
- Model/checkpoint policy: classification uses existing fixed round-25 endpoints; regression reuses the immutable R0-v2 source model and fixed A4 classifier, fits paired R83/R84 target models, and opens test only after locks; QC uses existing authoritative Q1 predictions and scores.
- Seeds: seed42 only; no multi-seed extension.
- DA / calibration / QC controls: target-information regime must be explicit for every classification row; Ridge alpha rule and grid are identical across budgets; source prior is immutable; QC score/policy is immutable and the main curve uses actual coverage.
- Primary regression scope: routed all-sample `S_ALL`; Oracle and route-correct scopes may be supplementary only if already free from the same prediction rows.
- Statistical scope: paired raw-file-grouped bootstrap describes test uncertainty conditional on the one frozen calibration subset; it is not calibration-subset or multi-seed uncertainty.

## Risks, unknowns, conflicts, and stopping rules

- Resolved amendment: the older 100%/50%/25% proposal is superseded for Experiment B by the later frozen 20%/15%/10%/5% manifest and explicit user approval. The old file remains read-only.
- The exact new runner/config paths and commit hash are `unknown` until registry and implementation; execution is blocked until they are registered.
- CORAL/DANN canonical endpoint inclusion requires hash and schema reconciliation with the existing comparator table; failure excludes the affected row rather than substituting a legacy result.
- The selected target alpha may change under the fixed grouped-CV rule; manual alpha equality or manual retuning is forbidden.
- Backend budget evidence cannot be written as end-to-end low-label or few-shot commissioning evidence.
- QC curve points are retrospective ranking diagnostics; only locked HC90/HC95 are operating points.
- No favorable numeric result is an acceptance condition. Protocol completeness, identity matching, provenance, and faithful reporting are the acceptance conditions.
- Any fail-closed condition in the design stops only the affected block and must be reported before scope changes are proposed.
