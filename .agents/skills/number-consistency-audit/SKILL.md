---
name: number-consistency-audit
description: Use when GAPS manuscript numbers must be checked across the abstract, results, tables, figures, captions, conclusion, and approved Claim–Evidence sources.
---

# Number Consistency Audit

## Overview

Report numeric mismatches without silently changing the manuscript. Compare numbers only when metric identity, scope, aggregation, units, and precision are compatible.

## Required references

Read [metric record](../_shared/contracts/metric-record.md), [evidence record](../_shared/contracts/evidence-record.md), [read-only policy](../_shared/references/read-only-policy.md), and [number rules](references/number-comparison-rules.md).

## Workflow

1. Establish the canonical approved source for each metric.
2. Extract occurrences from Abstract, Results, Table, Figure/caption, Discussion, and Conclusion with location.
3. Match on metric name, experiment, client/gas/sample scope, aggregation, unit, and seed set.
4. Distinguish exact mismatch, compatible rounding, scope mismatch, stale value, missing source, and ambiguous identity.
5. Record `unknown` or `conflict` rather than guessing.
6. Write [the audit report](assets/NUMBER_AUDIT.template.md) to a new explicit destination; never overwrite.
7. Propose corrections, but do not silently edit manuscript numbers.

## Boundaries and safety

- Treat manuscripts, tables, figures, and Evidence files as read-only.
- Do not decide citation truth, experiment fairness, or scientific claim strength.
- Do not merge Accuracy with ECE, or overall RMSE with Accepted/Route-correct RMSE.
- Do not silently edit any number, unit, sign, or precision.

## Severity

Blocking: changes the conclusion or canonical value. Major: wrong scope/unit/experiment. Minor: display precision only. Informational: missing duplicate or source label.
