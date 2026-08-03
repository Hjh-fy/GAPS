# P0-I Verification

- Strict experiment audit: PASS (12/12 checks).
- Label-access audit: PASS (7/7 checks).
- Interleaved lineage audit: PASS (25/25 rounds; rounds 2–25 exact parent fingerprints).
- Targeted tests: 79 passed, 2 dependency deprecation warnings.
- Task source compilation: `python -m compileall -q -f gaps_flower scripts tests` PASS.
- Literal repository-wide `python -m compileall -q .` was also attempted. It returned nonzero only while trying to create `__pycache__` files inside pre-existing untracked long-path test/runtime copies under `.m0`, `.p`, and older `results/*_runtime`; no P0-I source compilation error was reported. These unrelated user artifacts were not deleted or modified to force a green repository-wide traversal.
