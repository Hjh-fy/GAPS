import csv
from pathlib import Path


def _write_rows(path: Path, changed_qc: bool = False) -> None:
    fields = ["sample_index", "pred_class", "selected_profile", "qc_decision", "final_ppm"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1360):
            writer.writerow(
                {
                    "sample_index": index,
                    "pred_class": index % 4,
                    "selected_profile": "H8" if index % 4 == 1 else "direct",
                    "qc_decision": "reject" if changed_qc and index == 7 else "accept",
                    "final_ppm": f"{index / 10:.6f}",
                }
            )


def test_parity_rejects_one_qc_mismatch(tmp_path: Path) -> None:
    from scripts.validate_iotj_b5_c5_runtime_parity import validate_parity

    reference = tmp_path / "reference.csv"
    runtime = tmp_path / "runtime.csv"
    _write_rows(reference)
    _write_rows(runtime, changed_qc=True)

    report = validate_parity(reference, runtime)

    assert report["status"] == "failed"
    assert report["qc_decision_mismatches"] == 1


def test_parity_accepts_exact_1360_rows(tmp_path: Path) -> None:
    from scripts.validate_iotj_b5_c5_runtime_parity import validate_parity

    reference = tmp_path / "reference.csv"
    runtime = tmp_path / "runtime.csv"
    _write_rows(reference)
    _write_rows(runtime)

    report = validate_parity(reference, runtime)

    assert report["status"] == "equivalent"
    assert report["max_abs_ppm_delta"] == 0.0


def _write_c5_h8_rows(path: Path, *, reference: bool, changed_risk: bool = False, changed_auto: bool = False) -> None:
    ppm_field = "target_ridge_plus_source_preds_ppm" if reference else "h8_ppm"
    fields = ["sample_index", "pred_class", ppm_field, "deployment_risk_full", "qc_decision", "qc_workpoint"]
    if not reference:
        fields.append("auto_output_ppm")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(1360):
            risk = 0.5 + (0.1 if changed_risk and index == 7 else 0.0)
            row = {"sample_index": index, "pred_class": index % 4, ppm_field: index / 10, "deployment_risk_full": risk, "qc_decision": "accept", "qc_workpoint": "HC95"}
            if not reference:
                row["auto_output_ppm"] = index / 10 + (1.0 if changed_auto and index == 7 else 0.0)
            writer.writerow(row)


def test_c5_h8_parity_rejects_risk_and_auto_output_mismatch(tmp_path: Path) -> None:
    from scripts.validate_iotj_b5_c5_runtime_parity import validate_c5_h8_parity

    reference, runtime = tmp_path / "reference.csv", tmp_path / "runtime.csv"
    _write_c5_h8_rows(reference, reference=True)
    _write_c5_h8_rows(runtime, reference=False, changed_risk=True, changed_auto=True)

    report = validate_c5_h8_parity(reference, runtime, "HC95")

    assert report["status"] == "failed"
    assert report["risk_mismatches"] == report["auto_output_mismatches"] == 1


def test_c5_h8_parity_accepts_complete_stream_and_rejects_wrong_workpoint(tmp_path: Path) -> None:
    from scripts.validate_iotj_b5_c5_runtime_parity import validate_c5_h8_parity
    import pytest

    reference, runtime = tmp_path / "reference.csv", tmp_path / "runtime.csv"
    _write_c5_h8_rows(reference, reference=True)
    _write_c5_h8_rows(runtime, reference=False)
    assert validate_c5_h8_parity(reference, runtime, "HC95")["status"] == "equivalent"
    with pytest.raises(ValueError, match="unsupported"):
        validate_c5_h8_parity(reference, runtime, "FULL")
