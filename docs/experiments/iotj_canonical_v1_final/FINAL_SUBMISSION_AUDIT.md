# Final submission audit

Status: **EVIDENCE CLOSURE COMPLETE WITH BLOCKERS**.

Passed: frozen dataset hash; target-specific A4 checkpoint hashes; no target-test selection; R84 and QC provenance; 1,000-repeat same-budget random QC; no quality-based deletion; portable package preflight; exact Pi 5 package SHA; parameter-count semantics corrected; canonical 83D/84D comparison; complete FedAvg/FedProx/SCAFFOLD/MMD/A0T/GAPS comparator matrix; canonical SCAFFOLD sanity audit; completed strict raw-file-disjoint sensitivity.

Blockers/limitations: (1) strict non-overlap C5 loses 0.3005 Macro-F1 and adds 54.461 ppm S_ALL RMSE, triggering both preregistered collapse flags; (2) A4 versus equal-label A0T has mixed-sign, near-zero per-target Macro-F1 deltas at seed42, so no material classification-superiority claim is supported beyond label access; (3) canonical figures still need regeneration; (4) the requested manuscript v7 source was not available, so a six-way manuscript consistency PASS cannot be issued; (5) canonical calibration-budget sensitivity is absent.

The C5 methane 225 ppm repeat1 anomaly is retained: S_ALL RMSE 70.969 ppm versus 20.851 ppm for repeat2. No sample was deleted.
