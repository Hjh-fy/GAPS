"""Validate a rich-residual deployment bundle on target test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


CLASS_RANGES = {0: 112.5, 1: 225.0, 2: 112.5, 3: 225.0}
CO_CLASS = 1


def client_num(client: str) -> int:
    return int(str(client).upper().replace("CLIENT_", "").replace("C", ""))


def co_bin(true_ppm: float) -> str:
    if true_ppm <= 100.0:
        return "CO_low_25_100"
    if true_ppm <= 175.0:
        return "CO_mid_125_175"
    return "CO_high_200_250"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    pred = np.asarray([float(row[pred_key]) for row in rows], dtype=np.float64)
    true = np.asarray([float(row["true_ppm"]) for row in rows], dtype=np.float64)
    cls = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    if pred.size == 0:
        return {"N": 0, "RMSE": None, "MAE": None, "NRMSE": None, "Bias": None, "P90AE": None}
    err = pred - true
    ranges = np.asarray([CLASS_RANGES[int(c)] for c in cls], dtype=np.float64)
    return {
        "N": int(pred.size),
        "RMSE": float(np.sqrt(np.mean(err * err))),
        "MAE": float(np.mean(np.abs(err))),
        "NRMSE": float(np.sqrt(np.mean((err / ranges) ** 2))),
        "Bias": float(np.mean(err)),
        "P90AE": float(np.percentile(np.abs(err), 90)),
    }


def summarize(rows: list[dict[str, Any]], pred_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"pred_key": pred_key, "scope": "ALL", **metrics(rows, pred_key)}]
    for client in sorted({row["client"] for row in rows}, key=client_num):
        c_rows = [row for row in rows if row["client"] == client]
        out.append({"pred_key": pred_key, "scope": client, **metrics(c_rows, pred_key)})
        co_rows = [row for row in c_rows if int(row["true_class"]) == CO_CLASS]
        out.append({"pred_key": pred_key, "scope": f"{client}-CO", **metrics(co_rows, pred_key)})
        for name in ["CO_low_25_100", "CO_mid_125_175", "CO_high_200_250"]:
            b_rows = [row for row in co_rows if co_bin(float(row["true_ppm"])) == name]
            out.append({"pred_key": pred_key, "scope": f"{client}-{name}", **metrics(b_rows, pred_key)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rich residual deployment runtime.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--clients", default="C3,C4,C5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle = Path(args.bundle).resolve()
    sys.path.insert(0, str(bundle / "runtime_src"))
    from gaps_deploy.final_runtime import FinalDeployRuntime  # noqa: WPS433

    data_root = Path(args.data_root)
    out = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    for client in [item.strip() for item in args.clients.split(",") if item.strip()]:
        cid = client_num(client)
        cdir = data_root / f"client_{cid}"
        features = np.load(cdir / "test_features.npy").astype(np.float32)
        phases = np.load(cdir / "test_phase_labels.npy").astype(np.int64)
        cls = np.load(cdir / "test_classification_labels.npy").astype(np.int64)
        reg = np.load(cdir / "test_regression_labels.npy").astype(np.float32)
        meta_path = cdir / "test_experiment_info.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else []
        runtime = FinalDeployRuntime(bundle, client, device=args.device)
        pred_rows = runtime.predict_batch(features, phase=phases, metadata=metadata)
        for idx, row in enumerate(pred_rows):
            true_class = int(cls[idx])
            item = dict(row)
            item.update(
                {
                    "client": client,
                    "split": "test",
                    "sample_index": idx,
                    "true_class": true_class,
                    "true_ppm": float(reg[idx, true_class]),
                    "corrected_delta": float(row["co_corrected_ppm"]) - float(row["final_ppm"]),
                }
            )
            all_rows.append(item)
    summary_rows = []
    for key in ["final_ppm", "co_corrected_ppm"]:
        summary_rows.extend(summarize(all_rows, key))
    write_csv(out / "runtime_predictions.csv", all_rows)
    write_csv(out / "runtime_summary.csv", summary_rows)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "bundle": str(bundle),
                "data_root": str(data_root),
                "clients": args.clients,
                "outputs": {
                    "predictions": str(out / "runtime_predictions.csv"),
                    "summary": str(out / "runtime_summary.csv"),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote runtime validation to {out}")


if __name__ == "__main__":
    main()
