# C0 Experiment Audit

Status: **PASS — C0 completed; V1 interleaved retained**

## Source endpoint

- Exactly one C1/C2 Flower source trajectory completed 25/25 rounds with seed 42, one local epoch, batch size 32, and the frozen Adam configuration.
- The source API received no target features or target labels. Target test remained sealed.
- The fixed round-25 ordered state fingerprint is `09565570f0f4c6d3d6e96d89eb936f37f095a4d23f55a889829f327583397665`.
- The captured round-25 context contains both client prototype sets and semantic prototypes. Client residuals are unavailable, matching the audited C0-A interleaved reference; therefore `device_residual` remains inactive in both lifecycle variants.

## Launcher evidence correction

`SOURCE_LAUNCHER_FAILURE_AUDIT.json` records an early point-in-time inference before the detached controller finished. Later endpoint evidence proves that the original controller completed the sole scientific run. A second foreground invocation stopped before launch because execution artifacts already existed. `SOURCE_LAUNCHER_TIMELINE_CORRECTION.json` supersedes the preliminary failure interpretation without deleting historical evidence. No duplicate source training endpoint was created.

## Final adaptation and leakage gate

- C3, C4, and C5 independently reloaded the same original round-25 source state.
- Each target ran exactly 100 registered A4 adaptation steps with no hyperparameter or checkpoint search.
- All registered active losses ran 100/100 steps. `align_reg_legacy` was unavailable, `device_residual` was unavailable by audited parity, and target CE had configured weight zero.
- The sealed target tests were opened only after all three step-100 checkpoint markers, SHA256 checks, common source fingerprint checks, and loss-activity checks passed.

| Target | Final-adapt checkpoint SHA256 | Adaptation seconds | Final Macro-F1 | Interleaved Macro-F1 | Delta | Gate |
|---|---|---:|---:|---:|---:|---|
| C3 | `b052bb65f0b773bdf4f161833c23f3ecc14fdeec9a72c7bb6e17db6cf238f0c5` | 186.269 | 0.988769 | 0.998507 | -0.009738 | FAIL |
| C4 | `47d600c41fae1b2dad481de7b923fbe5be296b0cbda8e92dee19988df5a84521` | 183.767 | 0.988205 | 0.997794 | -0.009589 | FAIL |
| C5 | `cb992439c83d7b5a246a36412da8bcf3915de7eab7fcabe0167ac472de02fe3e` | 168.248 | 0.940642 | 0.994126 | -0.053484 | FAIL |

The preregistered noninferiority margin was -0.005 absolute Macro-F1 for every target. All three final-only branches failed. The frozen C0 decision is `V1_INTERLEAVED_RETAINED`; no rescue search or protocol change was performed.
