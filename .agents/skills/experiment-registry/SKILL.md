---
name: experiment-registry
description: Use when GAPS experiment IDs, configurations, splits, checkpoints, seeds, result paths, statuses, or provenance must be registered or reconciled.
---

# Experiment Registry

## Overview

Maintain traceable experiment identity without guessing semantics from paths. Produce candidate records until every critical field has a verified source.

## Required references

Read [experiment record](../_shared/contracts/experiment-record.md), [handoff protocol](../_shared/contracts/handoff-protocol.md), [read-only policy](../_shared/references/read-only-policy.md), and [GAPS taxonomy](../_shared/references/gaps-taxonomy.md).

## Workflow

1. Accept only user-confirmed values or values supported by config, manifest, git, or result metadata.
2. Check that `experiment_id` is unique and all canonical fields are present.
3. Record provenance per field; use `unknown` when evidence is absent.
4. Preserve disagreeing sources and set `status=conflict`.
5. Create a candidate row from [the registry template](assets/experiment_registry.template.csv).
6. Require an explicit destination; if it exists, provide an update proposal and do not overwrite.
7. Route completed records to `result-analysis` and unresolved comparability issues to `experiment-audit`.

## Boundaries and safety

- Treat existing registry and result assets as read-only.
- Do not infer split, DA, calibration, QC, model, or checkpoint identity from directory names.
- Do not compute metrics, decide fairness, or approve Evidence.
- Never overwrite or reuse an experiment ID.

## Common mistakes

| Mistake | Required correction |
|---|---|
| “Looks like strong DA” | Record `unknown` until config or manifest proves it. |
| One row covers multiple seeds | Use explicit seed-set semantics or separate records. |
| Result path exists, so run is approved | Mark completed; approval requires audit. |
