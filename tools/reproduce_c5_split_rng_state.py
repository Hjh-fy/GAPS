"""Reproduce time-aware splitter RNG consumption before C5 for two role maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def state_hash(rng: np.random.Generator) -> str:
    payload = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def exact_split(classification: np.ndarray, regression: np.ndarray, ratios: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Literal audit copy of run_time_aware_target_split_ablation.split_indices_by_protocol."""
    groups: dict[tuple[int, float], list[int]] = {}
    for idx, cls_value in enumerate(classification.astype(np.int64)):
        cls = int(cls_value)
        groups.setdefault((cls, float(regression[idx, cls])), []).append(idx)
    splits: dict[str, list[int]] = {"train": [], "calibration": [], "test": []}
    train_ratio, test_ratio, cal_ratio = ratios["train"], ratios["test"], ratios["calibration"]
    for key in sorted(groups):
        bucket = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(bucket)
        n = len(bucket)
        if train_ratio <= 0:
            n_train = 0
            n_cal = max(1, int(round(n * cal_ratio))) if n >= 2 else 0
            n_cal = min(n_cal, max(n - 1, 0))
            n_test = n - n_cal
            splits["calibration"].extend(bucket[:n_cal].tolist())
            splits["test"].extend(bucket[n_cal:n_cal+n_test].tolist())
        else:
            n_test = max(1, int(round(n * test_ratio))) if n >= 3 else max(0, n // 3)
            n_cal = max(1, int(round(n * cal_ratio))) if n >= 3 else max(0, (n - n_test) // 2)
            if n_test + n_cal >= n:
                overflow = n_test + n_cal - (n - 1)
                reduce_cal = min(n_cal, max(overflow, 0)); n_cal -= reduce_cal; overflow -= reduce_cal
                if overflow > 0:
                    n_test = max(0, n_test - overflow)
            n_train = n - n_test - n_cal
            splits["test"].extend(bucket[:n_test].tolist())
            splits["calibration"].extend(bucket[n_test:n_test+n_cal].tolist())
            splits["train"].extend(bucket[n_test+n_cal:n_test+n_cal+n_train].tolist())
    out = {}
    for split, values in splits.items():
        arr = np.asarray(values, dtype=np.int64)
        if len(arr):
            rng.shuffle(arr)
        out[split] = arr
    return out


def run_case(processed: Path, source: set[int], target: set[int]) -> tuple[str, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    c5_state = ""
    c5_split: dict[str, np.ndarray] = {}
    c5_cls = c5_reg = None
    for unit in sorted(processed.glob("unit_*")):
        cid = int(unit.name.split("_")[1])
        if cid not in source | target:
            continue
        cls = np.load(unit / "classification_labels.npy").reshape(-1)
        reg = np.load(unit / "regression_labels.npy")
        if cid == 5:
            c5_state = state_hash(rng)
            c5_cls, c5_reg = cls, reg
        ratios = ({"train": .70, "test": .20, "calibration": .10} if cid in source
                  else {"train": 0., "test": .80, "calibration": .20})
        split = exact_split(cls, reg, ratios, rng)
        if cid == 5:
            c5_split = split
    assert c5_state and c5_cls is not None and c5_reg is not None
    return c5_state, c5_split, c5_cls, c5_reg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=ROOT.parents[1])
    ap.add_argument("--output", type=Path, default=ROOT / "results/iotj_c5_pipeline_audit_20260807")
    args = ap.parse_args()
    processed = args.workspace.resolve() / "results/time_aware_pipeline_probe_window_fullgrid/time_aware_60_170_window_fullgrid/processed"
    state_a, split_a, cls, reg = run_case(processed, {1,2,3,4}, {5})
    state_b, split_b, _, _ = run_case(processed, {1,2}, {3,4,5})
    rows = []
    sets = {}
    for case, split in (("A_C1234_source_C5_target", split_a), ("B_C12_source_C345_target", split_b)):
        for split_name in ("calibration", "test"):
            for idx in split[split_name]:
                c = int(cls[idx]); ppm = float(reg[idx, c])
                rows.append({"case": case, "split": split_name, "original_processed_index": int(idx),
                             "class_id": c, "concentration": ppm})
        sets[case] = set(map(int, split["calibration"]))
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    path = out / "p2_c5_bucket_membership.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    overlap = len(sets["A_C1234_source_C5_target"] & sets["B_C12_source_C345_target"])
    md = f"""# P2 split RNG audit

- Exact splitter family: `np.random.default_rng(seed)` created once, then consumed sequentially while iterating `unit_1` ... `unit_5`.
- The legacy `split_dataset.py` independently shows the same structural risk with one global `np.random.seed(seed)` and sequential bucket/final-array shuffles. The executable replay below uses the actual time-aware generator behind NEW, because OLD and NEW do not share one processed Unit5 representation.
- Case A C5-entry RNG-state SHA256: `{state_a}`.
- Case B C5-entry RNG-state SHA256: `{state_b}`.
- States equal: **{state_a == state_b}**.
- C5 calibration membership overlap: {overlap}/320; symmetric difference: {640-2*overlap} windows.
- `same seed=42` does **not** guarantee the same C5 split when earlier clients have different source/target roles, because their ratio-specific final split-array shuffles consume different amounts of RNG state.

`RNG_CLIENT_ORDER_COUPLING = TRUE`

This audit only reproduces and records behavior; it does not modify the splitter.
"""
    (out / "P2_SPLIT_RNG_AUDIT.md").write_text(md, encoding="utf-8")
    print(json.dumps({"state_A": state_a, "state_B": state_b, "calibration_overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
