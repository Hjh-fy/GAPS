# Gate C Experiment Audit

## Verdict: PASS

- canonical-v1, H1, A0T/A4 classifier checkpoints, R84 model files, endpoint prediction files, and calibration locks passed SHA/provenance checks.
- A0T and A4 use byte-identical frozen C5 R84 model files.
- The calibration-only 4x4 matrix was locked before test diagnostic access.
- The test stage was read-only, paired by physical identity, and grouped-bootstrap resampled raw filenames.
- No classifier/R84/QC training, refitting, threshold selection, or hyperparameter search occurred.
