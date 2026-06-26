# QC Post-Hoc Reliability Report

QC is evaluated here as a deployment reliability layer after model-profile selection. These metrics must not be used to choose H2.3 versus H8+C4.

## ALL Scope

| profile | role | full_RMSE | accept_coverage | accept_RMSE | review_coverage | reject_coverage | accepted_review_RMSE | reject_RMSE | high_error_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H2.3 | balanced | 18.6217 | 0.4233 | 5.9183 | 0.3411 | 0.2356 | 9.0653 | 34.7193 | 0.9089 |
| H8+C4 | co_priority | 18.3041 | 0.4233 | 5.4832 | 0.3411 | 0.2356 | 8.0644 | 34.8035 | 0.9033 |

## Interpretation

- `full_RMSE` is the model-capability metric already used by the mainline selector.
- Accepted/review/reject subsets describe how QC routes outputs for deployment.
- `auto_output_ppm` should be interpreted as the automatic output only for accepted rows; review/reject predictions remain audit values.
