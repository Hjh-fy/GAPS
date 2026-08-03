"""Method-specific, fail-closed target information access policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ALL_FIELDS = frozenset({"x", "class", "phase", "concentration"})


class TargetTestLeakageError(RuntimeError):
    pass


class TargetInformationPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetInformationPolicy:
    method: str
    calibration_fields: frozenset[str]
    test_fields: frozenset[str] = frozenset({"x", "class"})


_POLICIES = {
    **{
        method: TargetInformationPolicy(method, frozenset({"x"}))
        for method in ("coral", "mmd", "dann", "e0")
    },
    "gaps": TargetInformationPolicy("gaps", frozenset({"x", "class", "phase"})),
    "a4": TargetInformationPolicy("a4", frozenset({"x", "class", "phase"})),
    "a5": TargetInformationPolicy("a5", frozenset({"x", "class", "phase"})),
    "a6": TargetInformationPolicy("a6", frozenset({"x", "class", "phase"})),
    **{
        method: TargetInformationPolicy(method, frozenset())
        for method in ("fedavg", "fedprox", "scaffold", "a0", "a1", "a2", "a3")
    },
}


def policy_for(method: str) -> TargetInformationPolicy:
    key = str(method).strip().lower()
    if key not in _POLICIES:
        raise TargetInformationPolicyError(f"Unknown target-information method: {method}")
    return _POLICIES[key]


class TargetAccessLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.events: list[dict] = []

    def _record(self, event: dict) -> None:
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def authorize(
        self,
        *,
        method: str,
        stage: str,
        split: str,
        fields: Iterable[str],
        purpose: str,
    ) -> None:
        requested = frozenset(str(field) for field in fields)
        invalid = requested - ALL_FIELDS
        if invalid:
            raise TargetInformationPolicyError(f"Unknown target fields: {sorted(invalid)}")
        policy = policy_for(method)
        split_key = str(split).lower()
        stage_key = str(stage).lower()
        if split_key == "test":
            allowed = stage_key == "final_evaluation" and requested <= policy.test_fields
            severity = "INFO" if allowed else "HARD_FAIL"
            reason = (
                "fixed_endpoint_final_evaluation"
                if allowed
                else "target_test_sealed_outside_final_evaluation"
            )
        elif split_key == "calibration":
            allowed = requested <= policy.calibration_fields
            severity = "INFO" if allowed else "HARD_FAIL"
            reason = (
                "method_specific_calibration_policy"
                if allowed
                else "requested_calibration_fields_not_registered"
            )
        else:
            allowed = False
            severity = "HARD_FAIL"
            reason = "unknown_target_split"
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": policy.method,
            "stage": stage_key,
            "split": split_key,
            "fields": sorted(requested),
            "purpose": str(purpose),
            "allowed": bool(allowed),
            "severity": severity,
            "reason": reason,
        }
        self._record(event)
        if not allowed and split_key == "test":
            raise TargetTestLeakageError(f"HARD_FAIL {reason}")
        if not allowed:
            raise TargetInformationPolicyError(f"HARD_FAIL {reason}")


@dataclass
class FinalEvaluationToken:
    method: str
    target: str
    completion_marker: Path
    consumed: bool = False

    def consume(self, target: str) -> None:
        if self.consumed:
            raise TargetTestLeakageError("HARD_FAIL final-evaluation token already consumed")
        if str(target).upper() != self.target.upper():
            raise TargetTestLeakageError("HARD_FAIL final-evaluation token target mismatch")
        self.consumed = True


def unlock_target_test_for_final_evaluation(
    method: str,
    target: str,
    completion_marker: str | Path,
    ledger: TargetAccessLedger,
) -> FinalEvaluationToken:
    marker = Path(completion_marker)
    if not marker.is_file():
        raise TargetTestLeakageError("HARD_FAIL fixed-endpoint completion marker missing")
    ledger.authorize(
        method=method,
        stage="final_evaluation",
        split="test",
        fields={"x", "class"},
        purpose="fixed_endpoint_classification_metrics",
    )
    return FinalEvaluationToken(method=str(method).lower(), target=str(target), completion_marker=marker)


class _TensorOnlyDataset(Dataset):
    def __init__(self, values: np.ndarray):
        self.values = torch.from_numpy(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


def load_target_calibration_x(
    client_dir: str | Path,
    *,
    method: str,
    ledger: TargetAccessLedger,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    ledger.authorize(
        method=method,
        stage="adaptation" if method.lower() in {"coral", "mmd", "dann"} else "diagnostic",
        split="calibration",
        fields={"x"},
        purpose="x_only_domain_alignment",
    )
    feature_path = Path(client_dir) / "calibration_features.npy"
    features = np.load(feature_path, allow_pickle=False)
    if features.ndim != 3 or tuple(features.shape[1:]) != (100, 8):
        raise RuntimeError(f"FAIL_CLOSED target calibration feature shape: {features.shape}")
    if not np.issubdtype(features.dtype, np.number) or not np.all(np.isfinite(features)):
        raise RuntimeError("FAIL_CLOSED invalid target calibration features")
    values = features.astype(np.float32, copy=False)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        _TensorOnlyDataset(values),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
    )
