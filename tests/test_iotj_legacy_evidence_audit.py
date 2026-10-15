from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.audit_iotj_legacy_evidence import (
    COMMON_COLUMNS,
    generate,
    macro_f1_from_confusion,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_macro_f1_from_confusion_is_exact_for_identity() -> None:
    assert macro_f1_from_confusion([[2, 0], [0, 3]]) == 1.0


def test_generate_refuses_nonempty_output(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="REFUSE_TO_OVERWRITE"):
        generate(tmp_path, tmp_path, output)


def test_committed_inventory_has_required_schema_and_unknown_regression_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output = repo_root / "docs/experiments/iotj_legacy_evidence_audit_20260803"
    expected = {
        "LEGACY_EVIDENCE_INVENTORY.md",
        "classification_cross_direction_summary.csv",
        "server_adaptation_component_summary.csv",
        "regression_cross_direction_summary.csv",
        "EXPERIMENT_AUDIT.md",
        "sha256_index.json",
    }
    assert {path.name for path in output.iterdir()} == expected

    for name in (
        "classification_cross_direction_summary.csv",
        "server_adaptation_component_summary.csv",
        "regression_cross_direction_summary.csv",
    ):
        with (output / name).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == COMMON_COLUMNS

    with (output / "regression_cross_direction_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all(row["nrmse_cc"] == "unknown" for row in rows)
    assert all(row["nrmse_all"] == "unknown" for row in rows)
    assert all(row["evaluation_replay_status"] == "NOT_RUN_NO_FROZEN_PREDICTION_ASSET" for row in rows)

    index = json.loads((output / "sha256_index.json").read_text(encoding="utf-8"))
    assert index["counts"] == {
        "classification_rows": 9,
        "component_rows": 5,
        "regression_rows": 6,
        "recomputed_regression_rows": 0,
        "unknown_regression_nrmse_rows": 6,
    }
    for artifact in index["generated_artifacts"]:
        path = repo_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
    history_root = Path(index["history_root"])
    for artifact in index["source_artifacts"]:
        path = history_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
