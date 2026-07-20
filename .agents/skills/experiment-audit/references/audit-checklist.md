# Audit Checklist

- Same research question and sample role.
- Same source/target clients and split protocol.
- Same dataset version and preprocessing.
- Intended model/checkpoint comparison, with all other factors held constant.
- Explicit DA, calibration, QC, routing, and sample scope.
- Required baselines, seeds, failed runs, and uncertainty present.
- No target-test use in tuning, calibration, selection, thresholds, or stopping.
- Result and metric provenance resolves to immutable inputs.
- Any exception is labeled and prevents unsupported causal wording.
