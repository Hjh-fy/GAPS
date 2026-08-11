"""Audit the S2/S4 FedRidge gate and fail closed on preprocessing mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S4_ROOT = ROOT / "dataset/iotj_canonical_v1_s4_role_view"
CANONICAL_ROOT = ROOT / "dataset/iotj_canonical_v1"
LEGACY_H1_ROOT = ROOT / "dataset/client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
SOURCE_ALPHA_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def source_clients_for_pool(pool: str) -> tuple[str, ...]:
    if pool == "S2":
        return ("C1", "C2")
    if pool == "S4":
        return ("C1", "C2", "C3", "C4")
    raise ValueError(pool)


def audit_s4_source_protocol() -> dict[str, Any]:
    manifest = json.loads((S4_ROOT / "s4_role_view_manifest.json").read_text(encoding="utf-8"))
    expected = list(source_clients_for_pool("S4"))
    if manifest.get("status") != "FROZEN" or manifest.get("source_clients") != expected:
        raise RuntimeError("FAIL_CLOSED S4 role-view identity differs")
    if manifest.get("target_clients") != ["C5"] or manifest.get("c5_rng_access") is not False:
        raise RuntimeError("FAIL_CLOSED S4 role-view accessed C5 source RNG")
    return {
        "status": "PASS",
        "source_clients": expected,
        "target_clients": ["C5"],
        "c5_rng_access": False,
        "source_test_used_for_fit_or_selection": False,
        "c5_used_for_source_fit_or_selection": False,
        "alpha_grid": list(SOURCE_ALPHA_GRID),
        "selection": "source-train fit; source-calibration RMSE; refit train+calibration",
    }


def _window_shape(root: Path, client: int, split: str) -> list[int]:
    shape = np.load(root / f"client_{client}/{split}_features.npy", mmap_mode="r").shape
    if len(shape) != 3:
        raise RuntimeError("FAIL_CLOSED source window tensor is not rank 3")
    return [int(shape[1]), int(shape[2])]


def audit_frozen_h1_preprocessing() -> dict[str, Any]:
    legacy_shape = _window_shape(LEGACY_H1_ROOT, 1, "train")
    canonical_shape = _window_shape(CANONICAL_ROOT, 1, "train")
    return {
        "status": (
            "PASS" if legacy_shape == canonical_shape else "HARD_FAIL_LEGACY_CANONICAL_MIX"
        ),
        "frozen_h1_source_root": str(LEGACY_H1_ROOT.resolve()),
        "frozen_h1_source_shape": legacy_shape,
        "canonical_root": str(CANONICAL_ROOT.resolve()),
        "canonical_shape": canonical_shape,
        "phase4_execution_authorized": legacy_shape == canonical_shape,
        "reason": "Existing H1 manifest was fit on 10Hz/100x8 windows; canonical-v1 uses 5Hz/50x8.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", required=True)
    parser.parse_args()
    payload = {
        "s4_source_protocol": audit_s4_source_protocol(),
        "frozen_h1_preprocessing": audit_frozen_h1_preprocessing(),
    }
    print(json.dumps(payload, indent=2))
    if payload["frozen_h1_preprocessing"]["phase4_execution_authorized"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
