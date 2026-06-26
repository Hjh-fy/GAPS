# C45 -> C123 Target Ridge Deployment Candidate

This directory exports the reverse-direction selected profile as a runtime-readable artifact.

## Artifact

- Export script: `export_target_ridge_profile_artifact.py`
- Artifact: `results/deployment_target_ridge_c45_c123_candidate_20260626/rich_residual_candidate.json`
- Direction: C45 -> C123
- Target clients: C1, C2, C3
- Selected profile: target Ridge direct for all target clients
- Model count: 12 per-client/per-gas Ridge heads
- Route rescue: disabled

## Runtime Equivalence

- Check script: `compare_target_ridge_profile_artifact.py`
- Equivalence output: `results/equivalence_target_ridge_c45_c123_candidate_20260626/equivalence_summary.json`
- Compared rows: 8040 target-test windows
- Expected column: `ridge_direct_ppm`
- Mismatches: 0
- Max absolute difference: 6.82e-13
- Mean absolute difference: 2.26e-14

## Interpretation

The reverse-direction target Ridge direct profile is now not only an analysis result.
It has a deployment/runtime artifact that reproduces the formal analysis predictions exactly up to floating-point noise.

This complements the C12 -> C345 runtime-validated H2.3/H8 candidates and supports the direction-specific profile-selection framing:

- C12 -> C345: H2.3 balanced mainline, H8 + formal C4 route rescue as CO-specialist candidate.
- C45 -> C123: target Ridge direct as the clean balanced mainline.
