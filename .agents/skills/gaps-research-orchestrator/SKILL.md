---
name: gaps-research-orchestrator
description: Use when the current GAPS research stage, largest Evidence Gap, blocked prerequisite, or next project research Skill must be identified and prioritized.
---

# Gaps Research Orchestrator

## Overview

Act as a research project router. Answer only: current stage, largest Evidence Gap, and next Skill.

## Required references

Read [handoff protocol](../_shared/contracts/handoff-protocol.md), [Skill boundaries](../_shared/references/skill-boundaries.md), [read-only policy](../_shared/references/read-only-policy.md), and [stage gates](references/stage-gates.md).

## Routing workflow

1. Inspect the existence, completeness, status, and unresolved gaps of standard artifacts.
2. Determine the current stage using documented gates, not optimistic interpretation.
3. Rank gaps by whether they block trustworthy Evidence or manuscript consistency.
4. Select one primary next Skill and list prerequisites.
5. Write [project status](assets/PROJECT_STATUS.template.md) and [next actions](assets/NEXT_ACTIONS.template.md) only to explicit new destinations; never overwrite.

## Routing table

| Evidence Gap | Next Skill |
|---|---|
| Hypothesis/baseline/ablation missing | `experiment-planner` |
| Experiment identity or provenance missing | `experiment-registry` |
| Confirmed results lack statistics | `result-analysis` |
| Split/checkpoint/seed/fairness conflict | `experiment-audit` |
| Audited Evidence lacks claim mapping | `claim-evidence` |
| Manuscript numbers disagree | `number-consistency-audit` |

## Boundaries and safety

- Treat all project artifacts as read-only and use `unknown` or `conflict` explicitly.
- Do not perform child Skill work.
- Do not analyze metrics, audit fairness, write claims, edit manuscripts, or resolve conflicting experiment semantics.
- Do not route to paper writing while blocking Evidence gaps remain.

## Mandatory response

1. **Current stage**
2. **Largest Evidence Gap**
3. **Next Skill**, reason, required inputs, and stop condition
