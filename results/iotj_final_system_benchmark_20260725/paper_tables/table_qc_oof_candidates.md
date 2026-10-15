| candidate | risk_components | OOF_Spearman | lowest_decile_RMSE | highest_decile_RMSE | tail_enrichment_ratio | HC95_accepted_RMSE | HC95_yield | HC90_accepted_RMSE | HC90_yield | risk_direction | tail_enrichment | selection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QC1 | confidence | 0.143022 | 71.0148 | 34.7558 | 1.21094 | 28.7241 | 0.953125 | 27.9479 | 0.9 | FAIL | PASS | NOT_SELECTED |
| QC2 | confidence + prototype/support distance | 0.188792 | 6.43346 | 35.2088 | 1.22672 | 28.1532 | 0.95 | 27.8848 | 0.9 | PASS | PASS | SELECTED |
| QC3 | confidence + prototype/support distance + regression consistency | 0.196639 | 5.64708 | 19.7936 | 0.689638 | 29.1778 | 0.95 | 29.5258 | 0.9 | FAIL | FAIL | NOT_SELECTED |

QC2 was selected using calibration OOF evidence only. QC3 failed tail enrichment. The C5 test set was not used for candidate selection.
