# Read-only Policy

Treat existing `results/`, Markdown, CSV, JSON, configuration, dataset, and checkpoint assets as read-only.

1. Require an explicit destination before creating an artifact instance.
2. If the destination exists, stop and offer a diff, update proposal, or new filename; never overwrite by default.
3. Do not load checkpoints for inference in version 1.
4. Do not infer experiment semantics from names or paths.
5. Use `unknown` for missing facts and `conflict` for traceable disagreement.
6. Produce candidate records and reports, not silent edits to canonical files.
