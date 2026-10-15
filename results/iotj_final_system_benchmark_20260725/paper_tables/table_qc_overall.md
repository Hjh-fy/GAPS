# Selective-output quality–coverage trade-off on the C5 target device

中文标题：C5目标设备上的选择性输出质量—覆盖率权衡

| runtime | regression_structure | QC_workpoint | total_N | accept_N | review_N | reject_N | accepted_yield | accepted_plus_review_yield | full_RMSE | full_MAE | accepted_RMSE | accepted_MAE | accepted_NRMSE | accepted_plus_review_RMSE | review_RMSE | reject_RMSE | misclassified_accept_N | misclassified_review_N | misclassified_reject_N | CO_N | CO_accepted_yield | CO_accepted_RMSE | CO_high_N | CO_high_accepted_yield | CO_high_accepted_RMSE | deployment_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V4 | B5 + H1/H2/H3 + C5 Ridge | NO_QC | 1360 | 1360 | 0 | 0 | 1 | 1 | 26.025 | 9.48601 | 26.025 | 9.48601 | 0.207293 | 26.025 |  |  | 27 | 0 | 0 | 340 | 1 | 22.9142 | 102 | 1 | 35.6278 | NO_QC_REFERENCE |
| V4 | B5 + H1/H2/H3 + C5 Ridge | HC95 | 1360 | 1323 | 33 | 4 | 0.972794 | 0.997059 | 26.025 | 9.48601 | 18.852 | 7.8996 | 0.142034 | 26.0036 | 116.349 | 32.462 | 12 | 12 | 3 | 340 | 0.967647 | 19.7767 | 102 | 0.970588 | 28.8039 | FORMAL_BASELINE |
| V4 | B5 + H1/H2/H3 + C5 Ridge | HC90 | 1360 | 1235 | 107 | 18 | 0.908088 | 0.986765 | 26.025 | 9.48601 | 15.8328 | 6.97563 | 0.11046 | 25.0716 | 70.6431 | 65.644 | 5 | 15 | 7 | 340 | 0.847059 | 19.7243 | 102 | 0.803922 | 30.7267 | FORMAL_BASELINE |
| V5 | B5 + Federated H1 + C5 105D Ridge | NO_QC | 1360 | 1360 | 0 | 0 | 1 | 1 | 25.649 | 9.38375 | 25.649 | 9.38375 | 0.204295 | 25.649 |  |  | 27 | 0 | 0 | 340 | 1 | 22.665 | 102 | 1 | 35.0212 | NO_QC_REFERENCE |
| V5 | B5 + Federated H1 + C5 105D Ridge | HC95 | 1360 | 1275 | 41 | 44 | 0.9375 | 0.967647 | 25.649 | 9.38375 | 13.9178 | 6.9283 | 0.0932414 | 16.6511 | 53.6243 | 109.734 | 6 | 5 | 16 | 340 | 0.858824 | 19.6774 | 102 | 0.784314 | 30.4522 | VALID_CANDIDATE_NOT_PROMOTED |
| V5 | B5 + Federated H1 + C5 105D Ridge | HC90 | 1360 | 1183 | 113 | 64 | 0.869853 | 0.952941 | 25.649 | 9.38375 | 12.7723 | 6.42216 | 0.0817851 | 15.3687 | 31.6404 | 95.8997 | 3 | 5 | 19 | 340 | 0.641176 | 20.8318 | 102 | 0.392157 | 40.0014 | VALID_CANDIDATE_NOT_PROMOTED |

Notes: Runtime v5 has lower accepted RMSE but lower accepted yield. Its HC90 CO yield and accepted-RMSE promotion guard failed. Runtime v4 therefore remains the formal baseline; runtime v5 QC2 is valid but not globally superior.
