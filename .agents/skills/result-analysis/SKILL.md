---
name: result-analysis
description: Use when confirmed GAPS CSV or JSON results require descriptive statistics, uncertainty, significance, effect sizes, anomaly checks, or paper table and figure proposals.
---

# Result Analysis

## Overview

Analyze only results whose experiment identity and metric schema are confirmed. Keep statistical findings separate from experimental fairness decisions.

## Required references

Read [metric record](../_shared/contracts/metric-record.md), [experiment record](../_shared/contracts/experiment-record.md), [read-only policy](../_shared/references/read-only-policy.md), and [statistical reporting](references/statistical-reporting.md).

## Workflow

1. Confirm experiment IDs, schema, metric direction, units, sample scope, and seed set.
2. Label copied values `reported` and values calculated now `recomputed`.
3. Report sample counts, mean, standard deviation, confidence interval, and effect size when supported.
4. Check assumptions before significance tests and document corrections for multiple comparisons.
5. Report anomalies without deleting observations.
6. Write findings into [the analysis template](assets/RESULT_ANALYSIS.template.md) at an explicit new destination.
7. Hand comparability questions to `experiment-audit`.

## Boundaries and safety

- Treat inputs as read-only and never overwrite reports.
- Use `unknown` for missing schema or units and `conflict` for inconsistent sources.
- Do not decide whether comparisons are fair.
- Do not invent seeds, recompute from screenshots, or promote findings to approved claims.

## Quick reference

| Need | Minimum output |
|---|---|
| Across seeds | n, mean, SD, CI |
| Group comparison | test, assumptions, exact p, effect size, CI |
| Anomaly | rule, affected record, sensitivity note |
