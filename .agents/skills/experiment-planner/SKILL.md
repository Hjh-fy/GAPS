---
name: experiment-planner
description: Use when a GAPS research question, hypothesis, baseline comparison, ablation, metric set, or expected Evidence must be fixed before running experiments.
---

# Experiment Planner

## Overview

Turn a research question into a controlled, reviewable experiment design. Freeze comparison semantics before execution; never infer a successful conclusion in advance.

## Required references

Read [experiment record](../_shared/contracts/experiment-record.md), [GAPS taxonomy](../_shared/references/gaps-taxonomy.md), [read-only policy](../_shared/references/read-only-policy.md), and [Skill boundaries](../_shared/references/skill-boundaries.md).

## Workflow

1. Confirm the research brief, hypothesis, source/target roles, split protocol, resource budget, and existing baselines.
2. Give each hypothesis an ID and a falsifiable statement.
3. Define controls, baselines, ablations, held-constant variables, seeds, metrics, slices, and acceptance criteria.
4. Map each planned comparison to Expected Evidence without inventing numeric results.
5. Mark missing facts `unknown` and contradictory inputs `conflict`; stop affected rows.
6. Instantiate [plan](assets/EXPERIMENT_PLAN.template.md), [matrix](assets/EXPERIMENT_MATRIX.template.csv), and [ablation plan](assets/ABLATION_PLAN.template.md) only at an explicit new destination.
7. Hand the confirmed design to `experiment-registry`.

## Output contract

- `EXPERIMENT_PLAN.md`: hypotheses, controls, acceptance criteria, budget, and risks.
- `EXPERIMENT_MATRIX.csv`: one canonical record per planned configuration.
- `ABLATION_PLAN.md`: factor, levels, held constants, Expected Evidence, and stopping rule.

## Boundaries and safety

- Treat existing experiment assets as read-only. Never overwrite an existing plan or matrix.
- Do not run experiments, register completed runs, analyze results, or claim support.
- Do not convert an `unknown` split, checkpoint, DA, calibration, or QC mode into a guess.
- Preserve every `conflict` and resolve it before execution.

## Common mistakes

| Mistake | Required correction |
|---|---|
| Mixing `7:2:1` and `8:2` rows | Use separate protocols. |
| Changing model and DA together | Add isolated ablations or label confounding. |
| Writing “will improve” | State a falsifiable hypothesis and Evidence criterion. |
| Treating one seed as stability Evidence | Plan repeated seeds or state the limitation. |
