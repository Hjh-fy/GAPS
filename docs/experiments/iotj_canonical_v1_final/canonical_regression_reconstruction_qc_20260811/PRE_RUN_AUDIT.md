# Canonical Regression Reconstruction + QC Necessity Pre-run Audit

Status: **PASS FOR IMPLEMENTATION-PLAN REVIEW; FORMAL EXECUTION NOT STARTED**

## Baseline and workspace

- Base commit: `26d453eed62057fff45cb6abdb96037b48112ba4`.
- Branch: `codex/iotj-final-classification-le1`.
- The prior regression/QC closure remains `HARD_FAIL_LEGACY_CANONICAL_MIX` and
  is not promoted into this study.
- Existing unrelated watcher logs and temporary pytest directories are outside
  this design commit and must remain untouched.

## Canonical asset readiness

- C1/C2 each provide canonical train, calibration, and test arrays with shape
  `N x 50 x 8`.
- C3/C4/C5 each provide formal canonical calibration and test arrays with shape
  `N x 50 x 8`.
- Calibration/test manifest hashes are frozen in `protocol_manifest.json`.
- The canonical aggregate hash is
  `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.

The target split remains fixed. Historical audits found raw-file/raw-time
relationships between some calibration and test windows. This study does not
repair or hide that limitation and cannot support a strict raw-file-disjoint
generalization claim.

## C0 reuse and new-run boundary

The official C3/C4/C5 interleaved endpoints and their test metrics exist and are
eligible for selective reuse after a runtime provenance audit. Their current
Macro-F1 values are 0.9985068849, 0.9977941081, and 0.9941260906.

The old G1 post-hoc A4 endpoint is not eligible for C0-B. Its manifest states
`interleaved_client_statistics_available: false`, and its loss-activity audit
shows unavailable prototype/consistency inputs. A new source trajectory that
preserves the exact round-25 client-derived adaptation inputs is required.

## H1 and 83D porting boundary

The inspected extractor implements 83 sensor descriptors and 21 metadata/phase
descriptors. It does not hard-code a 100-sample index or 10-Hz constant. Its
sample-difference and sample-slope descriptors are accepted only under the
approved fixed-5-Hz interpretation.

The whole extractor file SHA256 at design freeze is
`7627b72ee4e1823d24c374d41a6c931f66b5efedd6eaf4a839c62e7b5b1fa72a`.
Implementation must additionally fingerprint the exact function source and the
ordered 83D/104D feature-name lists.

No existing H1 or 83D cache has been approved for reuse. The default action is
recompute from canonical 50x8 arrays. Any discovered cache must independently
pass the canonical cache contract before reuse; otherwise it is rejected.

## FedRidge protocol readiness

The repository already contains a two-phase sufficient-statistics flow with:

- local feature moments;
- server global standardization;
- local normal equations;
- distributed validation SSE/count;
- an unregularized intercept; and
- a server API that rejects raw/sample-level inputs.

Only its mathematics and API separation may be reused. All numerical assets and
legacy alpha decisions must be regenerated from canonical-v1. Exact recovery is
required; practical-equivalence fallback is disallowed.

## QC availability boundary

The historical equal-mean QC depended on legacy source-prior ensemble assets.
This design does not authorize silently importing them or training additional
deep regression models. Q0 must therefore run an explicit canonical-input
availability audit before labeling any policy `EXISTING_EQUAL_MEAN_QC`.

If exact canonical reconstruction is unavailable, Q4 is marked unavailable and
Q1 may be triggered under the registered rule. This is a provenance result, not
a numerical failure of the historical policy.

## Pre-run verdict

The scientific questions, fixed inputs, decision thresholds, leakage gates,
conditional branches, and stop rules are sufficiently specified for an
implementation plan. No experiment, target-test evaluation, alpha selection,
QC fitting, or manuscript edit has occurred under this design commit.
