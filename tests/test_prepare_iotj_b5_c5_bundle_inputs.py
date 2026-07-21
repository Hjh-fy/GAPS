import csv
import hashlib
import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_reference(path: Path, *, workpoint: str = "HC90") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "pred_class", "qc_decision", "final_ppm", "qc_workpoint"],
        )
        writer.writeheader()
        for index in range(1360):
            writer.writerow(
                {
                    "sample_index": index,
                    "pred_class": index % 4,
                    "qc_decision": "accept",
                    "final_ppm": "12.5",
                    "qc_workpoint": workpoint,
                }
            )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    classifier = tmp_path / "classifier.pth"
    classifier.write_bytes(b"classifier")
    classifier_sha = hashlib.sha256(classifier.read_bytes()).hexdigest()
    r4 = tmp_path / "r4.json"
    _write_json(
        r4,
        {
            "classifier_sha256": classifier_sha,
            "source_aug_target_ridge_policy": {
                "feature_names": ["feature_a"],
                "switch_rule": {"class_ids": [0, 1, 2, 3]},
            },
        },
    )
    h23 = tmp_path / "h23.json"
    _write_json(h23, {"classifier_sha256": classifier_sha})
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    for name in ["risk_policy.json", "component_calibrator.json", "feature_reference.json", "risk_selection.json"]:
        _write_json(qc_dir / name, {})
    source = tmp_path / "hc90.csv"
    _source_reference(source)
    return {"classifier": classifier, "r4": r4, "h23": h23, "qc": qc_dir, "source": source}


def test_prepare_writes_bound_metadata_and_external_parity_reference(tmp_path: Path) -> None:
    from scripts.prepare_iotj_b5_c5_bundle_inputs import prepare_bundle_inputs

    paths = _inputs(tmp_path)
    result = prepare_bundle_inputs(
        classifier=paths["classifier"],
        r4_policy=paths["r4"],
        h23_reference=paths["h23"],
        qc_dir=paths["qc"],
        hc90_reference=paths["source"],
        output_dir=tmp_path / "prepared",
        frozen_commit="a" * 40,
        source_archive_sha256="b" * 64,
    )

    assert set(result) == {
        "classifier", "r4_policy", "h23_reference", "qc_risk_policy", "qc_component_calibrator",
        "qc_feature_reference", "qc_risk_selection", "feature_schema", "class_map", "normalization",
        "offline_reference_1360",
    }
    schema = json.loads((tmp_path / "prepared" / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["classifier"]["strict_load_verified"] is True
    assert schema["window"]["shape"] == [100, 8]
    with (tmp_path / "prepared" / "offline_reference_1360.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1360
    assert set(rows[0]) == {"sample_index", "pred_class", "selected_profile", "qc_decision", "final_ppm"}
    assert rows[0]["selected_profile"] == "b5_c5_r4_h23_hc90"


def test_prepare_rejects_policy_that_does_not_bind_classifier(tmp_path: Path) -> None:
    from scripts.prepare_iotj_b5_c5_bundle_inputs import prepare_bundle_inputs

    paths = _inputs(tmp_path)
    _write_json(paths["h23"], {"classifier_sha256": "0" * 64})

    with pytest.raises(ValueError, match="h23_reference classifier hash"):
        prepare_bundle_inputs(
            classifier=paths["classifier"], r4_policy=paths["r4"], h23_reference=paths["h23"],
            qc_dir=paths["qc"], hc90_reference=paths["source"], output_dir=tmp_path / "prepared",
            frozen_commit="a" * 40, source_archive_sha256="b" * 64,
        )


def test_prepare_rejects_non_hc90_parity_source(tmp_path: Path) -> None:
    from scripts.prepare_iotj_b5_c5_bundle_inputs import prepare_bundle_inputs

    paths = _inputs(tmp_path)
    _source_reference(paths["source"], workpoint="FULL")

    with pytest.raises(ValueError, match="HC90"):
        prepare_bundle_inputs(
            classifier=paths["classifier"], r4_policy=paths["r4"], h23_reference=paths["h23"],
            qc_dir=paths["qc"], hc90_reference=paths["source"], output_dir=tmp_path / "prepared",
            frozen_commit="a" * 40, source_archive_sha256="b" * 64,
        )
