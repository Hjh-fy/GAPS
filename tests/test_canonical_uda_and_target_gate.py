from __future__ import annotations

import inspect
from pathlib import Path

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
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.1, 0.9]])
    y = torch.tensor([0, 1, 0, 1])
    return DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)


def _target_loader() -> DataLoader:
    x = torch.tensor([[1.2, 0.1], [0.2, 1.1], [0.9, 0.3], [0.3, 0.8]])
    return DataLoader(TensorDataset(x), batch_size=2, shuffle=False)


def test_method_specific_target_information_policy() -> None:
    from gaps_flower.target_information import policy_for

    for method in ("coral", "mmd", "dann"):
        policy = policy_for(method)
        assert policy.calibration_fields == frozenset({"x"})
        assert policy.test_fields == frozenset({"x", "class"})
    gaps = policy_for("gaps")
    assert gaps.calibration_fields == frozenset({"x", "class", "phase"})
    assert "concentration" not in gaps.calibration_fields
    for method in ("fedavg", "fedprox", "scaffold", "a0", "a1", "a2", "a3"):
        assert policy_for(method).calibration_fields == frozenset()


@pytest.mark.parametrize("method", ["gaps", "a4", "a5", "a6"])
def test_gaps_runtime_authorization_records_exact_calibration_fields(
    tmp_path: Path, method: str
) -> None:
    from gaps_flower.target_information import (
        TargetAccessLedger,
        authorize_gaps_target_calibration,
    )

    ledger = TargetAccessLedger(tmp_path / f"{method}.jsonl")
    authorize_gaps_target_calibration(method=method, ledger=ledger)

    event = ledger.events[-1]
    assert event["stage"] == "adaptation"
    assert event["split"] == "calibration"
    assert event["fields"] == ["class", "phase", "x"]
    assert "concentration" not in event["fields"]
    assert event["allowed"] is True


def test_target_test_access_is_hard_fail_before_final_evaluation(tmp_path: Path) -> None:
    from gaps_flower.target_information import TargetAccessLedger, TargetTestLeakageError

    ledger = TargetAccessLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(TargetTestLeakageError, match="HARD_FAIL"):
        ledger.authorize(
            method="coral",
            stage="adaptation",
            split="test",
            fields={"x", "class"},
            purpose="training",
        )
    assert ledger.events[-1]["allowed"] is False
    assert ledger.events[-1]["severity"] == "HARD_FAIL"


def test_final_evaluation_token_requires_completion_and_is_single_use(tmp_path: Path) -> None:
    from gaps_flower.target_information import (
        TargetAccessLedger,
        TargetTestLeakageError,
        unlock_target_test_for_final_evaluation,
    )

    ledger = TargetAccessLedger(tmp_path / "ledger.jsonl")
    marker = tmp_path / "FCL-E2-CORAL-C3.completed.json"
    with pytest.raises(TargetTestLeakageError):
        unlock_target_test_for_final_evaluation("coral", "C3", marker, ledger)
    marker.write_text("{}", encoding="utf-8")
    token = unlock_target_test_for_final_evaluation("coral", "C3", marker, ledger)
    token.consume("C3")
    with pytest.raises(TargetTestLeakageError, match="already consumed"):
        token.consume("C3")


def test_calibration_x_loader_does_not_load_label_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gaps_flower.target_information import TargetAccessLedger, load_target_calibration_x

    client = tmp_path / "client_3"
    client.mkdir()
    np.save(client / "calibration_features.npy", np.zeros((3, 100, 8), dtype=np.float32))
    np.save(client / "calibration_classification_labels.npy", np.ones(3, dtype=np.int64))
    loaded: list[str] = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        loaded.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    ledger = TargetAccessLedger(tmp_path / "ledger.jsonl")
    loader = load_target_calibration_x(client, method="coral", ledger=ledger, batch_size=2)
    batch = next(iter(loader))

    assert isinstance(batch, torch.Tensor)
    assert loaded == ["calibration_features.npy"]
    assert ledger.events[-1]["fields"] == ["x"]


def test_calibration_x_loader_accepts_canonical_v1_50x8_windows(tmp_path: Path) -> None:
    from gaps_flower.target_information import TargetAccessLedger, load_target_calibration_x

    client = tmp_path / "client_3"
    client.mkdir()
    np.save(client / "calibration_features.npy", np.zeros((3, 50, 8), dtype=np.float32))
    ledger = TargetAccessLedger(tmp_path / "ledger.jsonl")

    batch = next(iter(load_target_calibration_x(client, method="mmd", ledger=ledger)))

    assert tuple(batch.shape) == (3, 50, 8)


@pytest.mark.parametrize("method", ["coral", "mmd", "dann"])
def test_canonical_uda_runs_exactly_one_unconditional_objective(method: str) -> None:
    from gaps_flower.canonical_uda import run_canonical_uda
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    model = TinyDomainModel()
    source_fingerprint = ordered_state_content_fingerprint(model.state_dict())
    _adapted, diagnostics, _seconds = run_canonical_uda(
        method,
        model,
        _source_loader(),
        _target_loader(),
        torch.device("cpu"),
        num_steps=2,
        model_lr=5e-4,
        alignment_weight=0.5,
        expected_source_fingerprint=source_fingerprint,
        formal=False,
    )

    assert len(diagnostics) == 2
    assert {row["method"] for row in diagnostics} == {method}
    assert all(row["alignment_scope"] == "unconditional_global" for row in diagnostics)
    assert all(row["target_fields"] == ["x"] for row in diagnostics)
    assert all(row["target_label_object_present"] is False for row in diagnostics)
    assert all(row["active_alignment"] == method for row in diagnostics)
    assert all(row["target_ce_status"] == "UNAVAILABLE" for row in diagnostics)
    assert all(row["class_conditional_status"] == "UNAVAILABLE" for row in diagnostics)
    assert all(row["pseudo_label_status"] == "DISABLED" for row in diagnostics)
    if method == "dann":
        assert all(row["dann_objective"] == "GRL_binary_BCE" for row in diagnostics)


def test_canonical_uda_api_has_no_target_label_parameter() -> None:
    from gaps_flower.canonical_uda import run_canonical_uda

    names = set(inspect.signature(run_canonical_uda).parameters)
    assert "target_x_loader" in names
    assert not any(name in names for name in ("target_y", "target_labels", "target_phase"))


def test_canonical_uda_rejects_wrong_source_fingerprint() -> None:
    from gaps_flower.canonical_uda import run_canonical_uda

    with pytest.raises(RuntimeError, match="source fingerprint mismatch"):
        run_canonical_uda(
            "coral",
            TinyDomainModel(),
            _source_loader(),
            _target_loader(),
            torch.device("cpu"),
            num_steps=1,
            expected_source_fingerprint="wrong",
            formal=False,
        )


def test_formal_canonical_uda_freezes_steps_lr_and_weight() -> None:
    from gaps_flower.canonical_uda import run_canonical_uda
    from gaps_flower.state_fingerprint import ordered_state_content_fingerprint

    model = TinyDomainModel()
    fingerprint = ordered_state_content_fingerprint(model.state_dict())
    with pytest.raises(ValueError, match="formal E2"):
        run_canonical_uda(
            "mmd",
            model,
            _source_loader(),
            _target_loader(),
            torch.device("cpu"),
            num_steps=99,
            expected_source_fingerprint=fingerprint,
            formal=True,
        )
