import csv
import json
from pathlib import Path

import numpy as np

from gaps_deploy.final_a4_runtime import FinalA4Runtime


def test_final_runtime_matches_frozen_first_row():
    root = Path("results/iotj_submission_evidence_closure_20260804/runtime_package")
    if not root.exists():
        return
    runtime = FinalA4Runtime(root)
    data = Path("dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid/client_5")
    window = np.load(data / "test_features.npy", mmap_mode="r")[0]
    phase = int(np.load(data / "test_phase_labels.npy", mmap_mode="r")[0])
    metadata = json.loads((data / "test_experiment_info.json").read_text(encoding="utf-8"))[0]
    observed = runtime.infer_one(window, metadata, phase)
    with Path("results/iotj_final_end_to_end_a4_20260804/qc/test_hc90_records.csv").open(encoding="utf-8", newline="") as handle:
        expected = next(csv.DictReader(handle))
    assert observed["pred_class"] == int(expected["pred_class"])
    for key in ("pred_83d_ppm", "pred_84d_h1_ppm", "classification_uncertainty_risk", "regression_disagreement_risk", "source_prior_disagreement_risk", "qc_risk_score_final"):
        assert abs(observed[key] - float(expected[key])) < 1e-6
    assert observed["accepted_hc90"] == int(expected["accepted"])
