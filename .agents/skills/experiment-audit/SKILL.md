---
name: experiment-audit
description: Use when GAPS experiments must be checked for completeness, fair comparison, reproducibility, split or checkpoint consistency, missing seeds or baselines, or data leakage risk.
---

# Experiment Audit

## Overview

Audit whether an experimental comparison can support Evidence. Preserve defects and conflicts; never repair them by changing results.

## Required references

Read [experiment record](../_shared/contracts/experiment-record.md), [evidence record](../_shared/contracts/evidence-record.md), [Skill boundaries](../_shared/references/skill-boundaries.md), and [audit checklist](references/audit-checklist.md).

## Workflow

1. Identify the planned claim and comparison set.
2. Check experiment IDs, source/target clients, split, model, checkpoint, DA, calibration, QC, seed, config, and result provenance.
3. Check baseline coverage, held constants, sample scope, leakage risk, and missing/failed runs.
4. Classify findings as blocking, major, minor, or informational.
5. Use `unknown` for absent evidence and `conflict` for disagreeing sources.
6. Record the verdict in [the audit template](assets/EXPERIMENT_AUDIT.template.md) without overwriting existing files.
7. Allow Evidence approval only when no blocking issue remains.

## Boundaries and safety

- Treat experimental assets as read-only.
- Require a new explicit report destination and never overwrite an existing audit.
- Do not recompute replacement metrics.
- Do not rerun experiments, change checkpoints, delete outliers, or write paper claims.
- A numerically favorable result does not override a fairness or leakage defect.

## Red flags

- Mixed `7:2:1` and `8:2` protocols.
- Different checkpoints or calibration/QC scopes in one comparison.
- Missing seeds or baseline rows.
- Target-test information used in selection, calibration, or threshold tuning.
