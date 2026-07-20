---
name: claim-evidence
description: Use when audited GAPS experimental or verified literature Evidence must be mapped to manuscript claims, claim strength, limitations, and source provenance.
---

# Claim Evidence

## Overview

Maintain a traceable path from approved Evidence to every manuscript claim. A claim may be weaker than its Evidence, never stronger.

## Required references

Read [evidence record](../_shared/contracts/evidence-record.md), [handoff protocol](../_shared/contracts/handoff-protocol.md), [Skill boundaries](../_shared/references/skill-boundaries.md), and [claim strength](references/claim-strength.md).

## Workflow

1. Assign stable claim and Evidence IDs.
2. Record the exact claim, scope, comparison, metric, source, manuscript locations, and limitations.
3. Verify experimental Evidence has audit status `approved`; keep unaudited Evidence as `draft` or `blocked`.
4. Rate support as direct, qualified, indirect, contradictory, or missing.
5. Use `unknown` for missing provenance and `conflict` for inconsistent sources.
6. Instantiate [the matrix](assets/CLAIMS_EVIDENCE.template.md) only at an explicit new destination and never overwrite.
7. Route prose drafting to `research-writing-skill` and citation entailment to a future citation audit.

## Boundaries and safety

- Treat source reports and manuscript files as read-only.
- Do not invent numbers, citations, mechanisms, or significance.
- Do not mark unaudited experimental Evidence as approved.
- Do not write the whole paper or resolve experiment fairness.

## Claim rule

Use the structure: claim → condition/scope → comparison → metric and uncertainty → source → limitation.
