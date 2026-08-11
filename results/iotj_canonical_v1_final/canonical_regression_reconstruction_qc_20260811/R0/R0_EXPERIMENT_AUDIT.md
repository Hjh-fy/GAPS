# R0 experiment audit

Verdict: **FAIL_CLOSED_EXACT_RECOVERY; downstream blocked**

- Frozen execution commit: `1b16f1e`.
- Dataset: canonical-v1, aggregate SHA256 `2f810d7e93cae5f361923184e9dc87d5ae59e0f59be9f52aff7e14f9f33e94f6`.
- Completed access: C1/C2 train and calibration arrays/labels only.
- Source alpha/model lock was persisted before any source test access.
- Source test labels were not opened; C3/C4/C5 caches and all target test labels were not opened.
- Exact recovery passed for Ethanol but failed at least one strict tolerance for CO, Ethylene, and Methane.
- The prediction tolerance passed for all gases, but the protocol forbids accepting practical equivalence alone.
- No feature, formula, solver, alpha grid, tolerance, or split changed; no rerun occurred.
- R1/R2/Q0/Q1 were not started.
