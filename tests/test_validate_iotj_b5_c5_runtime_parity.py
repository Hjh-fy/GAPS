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
