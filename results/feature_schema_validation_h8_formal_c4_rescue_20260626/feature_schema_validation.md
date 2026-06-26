# Feature Schema Validation

- bundle: `results\deployment_h8_formal_c4_rescue_candidate_20260625`
- data_root: `dataset\client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid`
- clients: `C3, C4, C5`
- status: **pass**

## Checks

| check | status | message |
| --- | --- | --- |
| C3_train_feature_shape | pass |  |
| C3_train_class_label_length | pass |  |
| C3_train_phase_label_length | pass |  |
| C3_train_regression_label_shape | pass |  |
| C3_train_metadata_length | pass |  |
| C3_calibration_feature_shape | pass |  |
| C3_calibration_class_label_length | pass |  |
| C3_calibration_phase_label_length | pass |  |
| C3_calibration_regression_label_shape | pass |  |
| C3_calibration_metadata_length | pass |  |
| C3_test_feature_shape | pass |  |
| C3_test_class_label_length | pass |  |
| C3_test_phase_label_length | pass |  |
| C3_test_regression_label_shape | pass |  |
| C3_test_metadata_length | pass |  |
| C4_train_feature_shape | pass |  |
| C4_train_class_label_length | pass |  |
| C4_train_phase_label_length | pass |  |
| C4_train_regression_label_shape | pass |  |
| C4_train_metadata_length | pass |  |
| C4_calibration_feature_shape | pass |  |
| C4_calibration_class_label_length | pass |  |
| C4_calibration_phase_label_length | pass |  |
| C4_calibration_regression_label_shape | pass |  |
| C4_calibration_metadata_length | pass |  |
| C4_test_feature_shape | pass |  |
| C4_test_class_label_length | pass |  |
| C4_test_phase_label_length | pass |  |
| C4_test_regression_label_shape | pass |  |
| C4_test_metadata_length | pass |  |
| C5_train_feature_shape | pass |  |
| C5_train_class_label_length | pass |  |
| C5_train_phase_label_length | pass |  |
| C5_train_regression_label_shape | pass |  |
| C5_train_metadata_length | pass |  |
| C5_calibration_feature_shape | pass |  |
| C5_calibration_class_label_length | pass |  |
| C5_calibration_phase_label_length | pass |  |
| C5_calibration_regression_label_shape | pass |  |
| C5_calibration_metadata_length | pass |  |
| C5_test_feature_shape | pass |  |
| C5_test_class_label_length | pass |  |
| C5_test_phase_label_length | pass |  |
| C5_test_regression_label_shape | pass |  |
| C5_test_metadata_length | pass |  |
| runtime_config_exists | pass | results\deployment_h8_formal_c4_rescue_candidate_20260625\runtime_config.json |
| runtime_input_shape | pass |  |
| norm_stats_mean | pass |  |
| norm_stats_std | pass |  |
| runtime_normalization | pass |  |
| runtime_client_package_C3 | pass | client_3_auto_v2_package |
| runtime_client_package_C4 | pass | client_4_auto_v2_package |
| runtime_client_package_C5 | pass | client_5_auto_v2_package |
| rich_residual_artifact_exists | pass | rich_residual_candidate.json |
| route_rescue_schema_v2 | pass | c4_route_rescue_policy.v2 |
| route_rescue_max_conf_margin | pass |  |

## Warnings

- C3 calibration: unexpected response_phase values ['pre_onset']
- C3 test: unexpected response_phase values ['pre_onset']
- C4 test: unexpected response_phase values ['pre_onset']
- C5 calibration: unexpected response_phase values ['pre_onset']
- C5 test: unexpected response_phase values ['pre_onset']
