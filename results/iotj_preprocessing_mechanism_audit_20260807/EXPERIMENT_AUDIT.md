# Experiment audit — preprocessing mechanism diagnostic

**Verdict: PASS for the stated diagnostic scope; BLOCKED for any formal preprocessing replacement.**

| Check | Status | Evidence |
|---|---|---|
| Existing assets immutable | PASS | all outputs confined to this directory |
| Same physical C3/C4/C5 membership | PASS | filename + window-start master keys |
| Classifier retraining | PASS | none performed |
| Target-test alpha/preprocessing selection | PASS | fixed Ridge grid selected only in calibration internal split; no canonical preprocessing selected |
| Formal-result replacement | BLOCKED | this audit is diagnostic-only |
| Time-bin coverage equality | MAJOR LIMITATION | N is reported in P6; invalid raw-missing windows are not silently filled |

This report does not approve changing the frozen formal preprocessing or `ceb6c78` evidence.
