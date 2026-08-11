from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ARRAY_SUFFIXES = (
    "features.npy",
    "classification_labels.npy",
    "regression_labels.npy",
    "phase_labels.npy",
)


def _write_split(root: Path, client: int, split: str, count: int) -> None:
    directory = root / f"client_{client}"
    directory.mkdir(parents=True, exist_ok=True)
    classes = np.asarray([index % 2 for index in range(count)], dtype=np.int64)
    concentrations = np.asarray([10.0 + 10.0 * ((index // 2) % 2) for index in range(count)])
    features = np.arange(count * 3 * 2, dtype=np.float32).reshape(count, 3, 2) + client * 1000
    regression = np.zeros((count, 4), dtype=np.float32)
    regression[np.arange(count), classes] = concentrations
    np.save(directory / f"{split}_features.npy", features)
    np.save(directory / f"{split}_classification_labels.npy", classes)
    np.save(directory / f"{split}_regression_labels.npy", regression)
    np.save(directory / f"{split}_phase_labels.npy", np.arange(count, dtype=np.int64) % 3)
    rows = [
        {
            "physical_identity": f"C{client}|{split}|{index}",
            "client_id": client,
            "class_id": int(classes[index]),
            "classification_label": int(classes[index]),
            "concentration": float(concentrations[index]),
            "regression_label": regression[index].tolist(),
            "phase_label": int(index % 3),
            "role": split,
        }
        for index in range(count)
    ]
    (directory / f"{split}_experiment_info.json").write_text(json.dumps(rows), encoding="utf-8")


def _canonical_fixture(root: Path) -> Path:
    for client in (1, 2):
        for split, count in (("train", 16), ("calibration", 8), ("test", 8)):
            _write_split(root, client, split, count)
    for client in (3, 4, 5):
        _write_split(root, client, "calibration", 8)
        _write_split(root, client, "test", 24)
    (root / "canonical_preprocessing_manifest.json").write_text(
        json.dumps({"candidate_id": "HZ5_MEAN_W10S", "seed": 42}), encoding="utf-8"
    )
    return root


def _ids(directory: Path, split: str) -> set[str]:
    rows = json.loads((directory / f"{split}_experiment_info.json").read_text(encoding="utf-8"))
    return {str(row["physical_identity"]) for row in rows}


def test_s4_role_view_preserves_c1_c2_and_c5_byte_content(tmp_path: Path) -> None:
    from tools.build_iotj_canonical_v1_s4_role_view import build_role_view, sha256_file

    canonical = _canonical_fixture(tmp_path / "canonical")
    output = tmp_path / "s4"
    build_role_view(canonical, output, seed=42)

    for client in (1, 2, 5):
        for source in (canonical / f"client_{client}").iterdir():
            if source.is_file():
                target = output / f"client_{client}" / source.name
                assert target.is_file()
                assert sha256_file(source) == sha256_file(target)


def test_s4_role_view_partitions_added_sources_without_overlap(tmp_path: Path) -> None:
    from tools.build_iotj_canonical_v1_s4_role_view import build_role_view

    canonical = _canonical_fixture(tmp_path / "canonical")
    output = tmp_path / "s4"
    manifest = build_role_view(canonical, output, seed=42)

    for client in (3, 4):
        source_pool = _ids(canonical / f"client_{client}", "calibration") | _ids(
            canonical / f"client_{client}", "test"
        )
        split_ids = [_ids(output / f"client_{client}", split) for split in ("train", "calibration", "test")]
        assert set.union(*split_ids) == source_pool
        assert not (split_ids[0] & split_ids[1])
        assert not (split_ids[0] & split_ids[2])
        assert not (split_ids[1] & split_ids[2])
        assert all(split_ids)
        assert manifest["clients"][f"C{client}"]["source_pool_count"] == 32


def test_s4_role_view_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    from tools.build_iotj_canonical_v1_s4_role_view import build_role_view

    canonical = _canonical_fixture(tmp_path / "canonical")
    first = build_role_view(canonical, tmp_path / "first", seed=42)
    second = build_role_view(canonical, tmp_path / "second", seed=42)
    assert first["partition_identity_sha256"] == second["partition_identity_sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        build_role_view(canonical, tmp_path / "first", seed=42)


def test_gate_a_commands_are_four_source_only_fixed_endpoint() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import build_gate_a_commands

    for method in ("fedavg", "gaps_dg_p"):
        commands = build_gate_a_commands(method)
        assert set(commands["clients"]) == {"C1", "C2", "C3", "C4"}
        joined = " ".join(commands["server"] + sum(commands["clients"].values(), []))
        assert "client_5" not in joined
        assert commands["protocol"]["target_access"] == "NONE"
        assert commands["protocol"]["rounds"] == 25
        assert commands["protocol"]["local_epochs"] == 1
        assert commands["protocol"]["seed"] == 42
        assert commands["protocol"]["checkpoint_selection"] == "fixed_round_25"
        assert commands["protocol"]["experiment_id"].startswith("CAN-V1-MB-A-S4-")
        assert commands["protocol"]["dataset_aggregate_sha256"] == "843459df765eb1525a30b91024e6db6a66a740858d0761ff84e67decb92389fa"


def test_gate_a_dg_changes_only_exact_g2_mechanism() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import build_gate_a_commands

    fedavg = build_gate_a_commands("fedavg")["protocol"]
    dg = build_gate_a_commands("gaps_dg_p")["protocol"]
    assert fedavg["optimizer"] == dg["optimizer"] == "Adam"
    assert fedavg["optimizer_lr"] == dg["optimizer_lr"] == 5e-4
    assert fedavg["prototype_alignment"] is False
    assert dg["prototype_alignment"] is True
    assert dg["lambda_proto"] == 0.05
    assert dg["replay"] is False
    assert dg["selective_aggregation"] is False
    assert dg["server_domain_adaptation"] is False


def test_gate_a_decision_supports_diversity_without_dg() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import decide_gate_a

    result = decide_gate_a(
        s2_fedavg_c5_f1=0.40,
        s2_dg_c5_f1=0.39,
        s4_fedavg_c5_f1=0.45,
        s4_dg_c5_f1=0.451,
        s4_fedavg_source_f1=0.99,
        s4_dg_source_f1=0.989,
    )
    assert result["source_diversity"] == "SOURCE_DIVERSITY_SUPPORTED"
    assert result["dg_mechanism"] == "DG_MECHANISM_NOT_SUPPORTED"
    assert result["next_action"] == "STOP_DG_EXPANSION"


def test_gate_a_decision_allows_only_proposal_for_promising_dg() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import decide_gate_a

    result = decide_gate_a(
        s2_fedavg_c5_f1=0.40,
        s2_dg_c5_f1=0.39,
        s4_fedavg_c5_f1=0.45,
        s4_dg_c5_f1=0.465,
        s4_fedavg_source_f1=0.99,
        s4_dg_source_f1=0.985,
    )
    assert result["dg_mechanism"] == "SOURCE_DG_PROMISING"
    assert result["next_action"] == "CREATE_MULTI_SEED_PROPOSAL_ONLY"


def test_gate_a_decision_retires_when_s4_does_not_improve() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import decide_gate_a

    result = decide_gate_a(
        s2_fedavg_c5_f1=0.40,
        s2_dg_c5_f1=0.39,
        s4_fedavg_c5_f1=0.405,
        s4_dg_c5_f1=0.395,
        s4_fedavg_source_f1=0.99,
        s4_dg_source_f1=0.99,
    )
    assert result["source_diversity"] == "SOURCE_DIVERSITY_NOT_SUPPORTED"
    assert result["dg_mechanism"] == "SOURCE_DG_RETIRED"
    assert result["next_action"] == "STOP_DG_EXPANSION"


def test_gate_a_reuse_audit_rejects_target_access_or_wrong_checkpoint(tmp_path: Path) -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import audit_s2_reuse

    checkpoint = tmp_path / "endpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    protocol = {
        "dataset": "iotj_canonical_v1",
        "rounds": 25,
        "local_epochs": 1,
        "seed": 42,
        "target_x": False,
        "target_y": False,
        "checkpoint_selection": "fixed_round_25",
    }
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "wrong",
        "protocol": protocol,
        "target_test_opened": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        audit_s2_reuse(path)

    from hashlib import sha256

    manifest["checkpoint_sha256"] = sha256(b"checkpoint").hexdigest()
    manifest["protocol"]["target_x"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target access"):
        audit_s2_reuse(path)


def test_gate_a_execution_roles_colocate_only_added_sources_on_ecs() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import execution_role_hosts

    assert execution_role_hosts() == {
        "server": "ecs",
        "C1": "pi",
        "C2": "c2",
        "C3": "ecs",
        "C4": "ecs",
    }


def test_gate_a_representation_diagnostic_uses_all_source_pairs() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import representation_diagnostics

    records = []
    for client, offset in ((1, 0.0), (2, 2.0), (3, 4.0)):
        for class_id, center in ((0, 0.0), (1, 10.0)):
            records.append(
                {
                    "client": client,
                    "class_id": class_id,
                    "phase_id": 0,
                    "feature": np.asarray([center + offset, 0.0]),
                }
            )
    result = representation_diagnostics(records, method="fixture", source_count=3)
    assert result["inter_source_pairs"] == 3
    assert result["within_class_inter_source_centroid_distance_mean"] == pytest.approx(8.0 / 3.0)
    assert result["between_class_centroid_margin_mean"] == pytest.approx(10.0)
    assert result["class_phase_cells"] == 2


def test_gate_a_requires_both_s4_locks_before_test_open(tmp_path: Path) -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import verify_s4_endpoint_locks

    for method in ("fedavg", "gaps_dg_p"):
        directory = tmp_path / method
        directory.mkdir()
        checkpoint = directory / "server_latest.pth"
        checkpoint.write_bytes(method.encode())
        manifest = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": __import__("hashlib").sha256(method.encode()).hexdigest(),
            "target_test_opened": False,
            "protocol": {"checkpoint_selection": "fixed_round_25"},
        }
        (directory / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "fixed_endpoint_complete.json").write_text(
            json.dumps({"fixed_endpoint": {"round": 25}, "target_test_opened": False}),
            encoding="utf-8",
        )
    locked = verify_s4_endpoint_locks(tmp_path)
    assert set(locked) == {"fedavg", "gaps_dg_p"}
    assert all(Path(value["checkpoint"]).is_file() for value in locked.values())

    bad = tmp_path / "gaps_dg_p/fixed_endpoint_complete.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["target_test_opened"] = True
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="endpoint lock"):
        verify_s4_endpoint_locks(tmp_path)


def test_gate_a_per_class_rows_can_be_reconstructed_from_frozen_predictions() -> None:
    from scripts.run_iotj_method_breakthrough_gate_a import per_class_rows_from_predictions

    rows = []
    for class_id in range(4):
        probs = [0.0] * 4
        probs[class_id] = 1.0
        rows.append({"true_class": class_id, **{f"prob_{i}": probs[i] for i in range(4)}})
    result = per_class_rows_from_predictions("fixture", rows)
    assert [row["gas"] for row in result] == ["Ethanol", "CO", "Ethylene", "Methane"]
    assert all(row["precision"] == 1.0 and row["recall"] == 1.0 and row["f1"] == 1.0 for row in result)
