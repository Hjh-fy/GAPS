from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TinyDomainModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(2, 4)
        self.classifier = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        return self.classifier(feat), feat, feat


def _source_loader() -> DataLoader:
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.1, 0.9]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _target_loader() -> DataLoader:
    x = torch.tensor(
        [[1.2, 0.1], [0.2, 1.1], [0.9, 0.3], [0.3, 0.8]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)


def _run(method: str):
    from gaps_flower.labeled_conditional_da import run_labeled_conditional_da
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    model = TinyDomainModel()
    original = copy.deepcopy(model.state_dict())
    fingerprint = ordered_state_content_fingerprint(model.state_dict())
    adapted, diagnostics, _seconds = run_labeled_conditional_da(
        method,
        model,
        _source_loader(),
        _target_loader(),
        torch.device("cpu"),
        num_steps=2,
        expected_source_fingerprint=fingerprint,
        formal=False,
    )
    return model, adapted, original, diagnostics


def test_registered_matrix_contains_only_the_six_approved_methods() -> None:
    from gaps_flower.labeled_conditional_da import registered_method_specs

    specs = registered_method_specs()
    assert list(specs) == [
        "coral_tce",
        "mmd_tce",
        "dann_tce",
        "cc_coral",
        "class_mmd",
        "cc_dann",
    ]
    assert all(spec["steps"] == 100 for spec in specs.values())
    assert all(spec["optimizer"] == "Adam" for spec in specs.values())
    assert all(spec["optimizer_lr"] == 5e-4 for spec in specs.values())
    assert all(spec["alignment_weight"] == 0.5 for spec in specs.values())
    assert [specs[key]["target_ce_weight"] for key in list(specs)[:3]] == [1.0] * 3
    assert [specs[key]["target_ce_weight"] for key in list(specs)[3:]] == [0.0] * 3


@pytest.mark.parametrize("method", ["coral_tce", "mmd_tce", "dann_tce"])
def test_tce_variants_activate_target_ce_and_global_alignment(method: str) -> None:
    _model, _adapted, _original, rows = _run(method)

    assert len(rows) == 2
    assert all(row["target_ce_active"] is True for row in rows)
    assert all(row["target_ce_weight"] == 1.0 for row in rows)
    assert all(row["class_conditional_active"] is False for row in rows)
    assert all(row["alignment_scope"] == "unconditional_global" for row in rows)
    assert all(row["target_fields"] == ["x", "class"] for row in rows)
    assert all(row["raw_target_ce"] > 0.0 for row in rows)


@pytest.mark.parametrize("method", ["cc_coral", "class_mmd", "cc_dann"])
def test_class_conditional_variants_use_labels_only_for_grouping(method: str) -> None:
    _model, _adapted, _original, rows = _run(method)

    assert len(rows) == 2
    assert all(row["target_ce_active"] is False for row in rows)
    assert all(row["target_ce_weight"] == 0.0 for row in rows)
    assert all(row["raw_target_ce"] == 0.0 for row in rows)
    assert all(row["class_conditional_active"] is True for row in rows)
    assert all(row["alignment_scope"] == "true_class_conditional" for row in rows)
    assert all(row["valid_class_count"] == 2 for row in rows)


def test_cc_dann_no_shared_class_returns_connected_zero() -> None:
    from gaps_flower.labeled_conditional_da import (
        SharedDomainDiscriminator,
        class_conditional_alignment,
    )

    source = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 4, requires_grad=True)
    source_y = torch.zeros(3, dtype=torch.long)
    target_y = torch.ones(3, dtype=torch.long)
    discriminator = SharedDomainDiscriminator(4)

    loss, valid = class_conditional_alignment(
        "cc_dann", source, source_y, target, target_y, discriminator
    )
    loss.backward()

    assert valid == []
    assert loss.item() == 0.0
    assert source.grad is not None
    assert target.grad is not None


def test_each_run_copies_but_does_not_mutate_the_source_checkpoint() -> None:
    model, adapted, original, _rows = _run("coral_tce")

    assert all(torch.equal(model.state_dict()[key], value) for key, value in original.items())
    assert any(
        not torch.equal(adapted.state_dict()[key], value)
        for key, value in original.items()
    )


def test_target_training_api_rejects_phase_or_concentration_objects() -> None:
    from gaps_flower.labeled_conditional_da import run_labeled_conditional_da
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    model = TinyDomainModel()
    fingerprint = ordered_state_content_fingerprint(model.state_dict())
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 0, 1])
    forbidden = DataLoader(
        TensorDataset(x, y, torch.zeros(4), torch.zeros(4)),
        batch_size=4,
    )
    with pytest.raises(RuntimeError, match="exactly x and class"):
        run_labeled_conditional_da(
            "cc_coral",
            model,
            _source_loader(),
            forbidden,
            torch.device("cpu"),
            num_steps=1,
            expected_source_fingerprint=fingerprint,
            formal=False,
        )


def test_formal_protocol_rejects_any_hyperparameter_change() -> None:
    from gaps_flower.labeled_conditional_da import run_labeled_conditional_da
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    model = TinyDomainModel()
    fingerprint = ordered_state_content_fingerprint(model.state_dict())
    with pytest.raises(ValueError, match="formal labeled-conditional DA"):
        run_labeled_conditional_da(
            "class_mmd",
            model,
            _source_loader(),
            _target_loader(),
            torch.device("cpu"),
            num_steps=99,
            expected_source_fingerprint=fingerprint,
            formal=True,
        )


def test_target_calibration_loader_reads_only_x_and_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.run_iotj_c5_labeled_conditional_da import load_target_calibration_xy

    client = tmp_path / "client_5"
    client.mkdir()
    np.save(client / "calibration_features.npy", np.zeros((8, 100, 8), dtype=np.float32))
    np.save(client / "calibration_classification_labels.npy", np.arange(8) % 4)
    np.save(client / "calibration_phase_labels.npy", np.zeros(8, dtype=np.int64))
    np.save(client / "calibration_regression_labels.npy", np.zeros((8, 4)))
    loaded: list[str] = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    loader, summary = load_target_calibration_xy(client, batch_size=4, seed=42)
    batch = next(iter(loader))

    assert loaded == [
        "calibration_features.npy",
        "calibration_classification_labels.npy",
    ]
    assert len(batch) == 2
    assert summary["fields"] == ["x", "class"]
    assert summary["num_examples"] == 8


def test_target_test_release_requires_all_six_completion_markers(tmp_path: Path) -> None:
    from gaps_flower.labeled_conditional_da import registered_method_specs
    from scripts.run_iotj_c5_labeled_conditional_da import require_all_endpoints

    methods = list(registered_method_specs())
    for method in methods[:-1]:
        directory = tmp_path / method
        directory.mkdir()
        (directory / "fixed_endpoint_complete.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="all six fixed endpoints"):
        require_all_endpoints(tmp_path, methods)
    final = tmp_path / methods[-1]
    final.mkdir()
    (final / "fixed_endpoint_complete.json").write_text("{}", encoding="utf-8")
    assert require_all_endpoints(tmp_path, methods) == methods


def test_protocol_manifest_binds_original_round25_source_checkpoint(tmp_path: Path) -> None:
    from scripts.run_iotj_c5_labeled_conditional_da import build_protocol_manifest

    checkpoint = tmp_path / "source.pth"
    checkpoint.write_bytes(b"round25-source")
    manifest = build_protocol_manifest(
        source_checkpoint=checkpoint,
        source_fingerprint="f" * 64,
        data_root=tmp_path / "dataset",
    )

    assert manifest["source_checkpoint_sha256"] == (
        "143ce017687837a8ac855c11354964825d6c8c646dcd4ada31791a803730848c"
    )
    assert manifest["source_checkpoint_ordered_fingerprint"] == "f" * 64
    assert manifest["target"] == "C5"
    assert manifest["calibration_examples"] == 320
    assert manifest["test_examples"] == 1360
    assert manifest["target_test_selection"] is False


def test_direct_script_entrypoint_can_inspect_without_pythonpath() -> None:
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/run_iotj_c5_labeled_conditional_da.py"),
            "inspect",
        ],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout
    assert '"source_checkpoint_exists": true' in completed.stdout.lower()
