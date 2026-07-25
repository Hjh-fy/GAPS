# Runtime v5 QC closure implementation plan

1. Freeze protocol amendment v2, input identities, historical HC semantics, and six v4
   hashes in the independent 20260725 result root.
2. Add failing unit tests for group isolation, deterministic folds, fold-local
   references/scales, candidate formulas, lock validation, fail-closed bundle
   loading, decisions, and runtime parity fields.
3. Implement reusable v5 QC policy primitives and the two-phase calibration/test
   evaluator.
4. Generate five-fold OOF rows, audits, candidate summaries, selection, full-320
   policy assets, and the immutable selection lock.
5. Validate the lock and evaluate the frozen 1360-row test exactly once; compute
   overall/per-gas/decile/CO-high diagnostics and v4 guards.
6. Build the QC-enabled Runtime v5 candidate and verify 320/1360 offline-runtime
   parity.
7. Run focused and related tests, audit leakage/provenance/frozen hashes, register
   the evidence, produce the Q10 decision, commit lightweight artifacts, and push.
