from scripts.build_iotj_canonical_v1_submission_bundle import REQUIRED_EVIDENCE_FILES


def test_submission_bundle_declares_all_required_evidence_files() -> None:
    assert REQUIRED_EVIDENCE_FILES == (
        "01_dataset_manifest.json",
        "02_final_experiment_state.json",
        "03_classification_final.csv",
        "04_regression_final.csv",
        "05_qc_final.csv",
        "06_quality_robustness.csv",
        "07_pi5_benchmark.csv",
        "08_model_size.json",
        "09_a0t_equal_label.csv",
        "10_fedridge_83d_84d.csv",
        "11_window_overlap.csv",
        "12_reproducibility_manifest.json",
    )
