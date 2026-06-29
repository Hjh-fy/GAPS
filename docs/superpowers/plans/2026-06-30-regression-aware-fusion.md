# Regression-Aware Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export official F6 r25 classification-backbone features, run H2.3 feature-fusion ablations, and produce an initial result report that says whether backbone embeddings help target ppm regression.

**Architecture:** Add one feature-export script and one Ridge ablation script. The export script reuses the Flower checkpoint loader and writes key-aligned CSVs for calibration/test splits. The ablation script reuses existing rich-stat, calibration split, Ridge fitting, and metric helpers, then compares feature groups under the official r25 checkpoint.

**Tech Stack:** Python, PyTorch, NumPy, existing `gaps_flower.evaluate_checkpoint`, existing `run_regression_head_ablation`, existing formal target Ridge helpers, pytest.

---

## File Structure

- Create `export_backbone_features.py`
  - Loads a Flower classification checkpoint.
  - Iterates C3/C4/C5 calibration/test splits in deterministic sample order.
  - Writes `backbone_features_calibration.csv`, `backbone_features_test.csv`, and `manifest.json`.
  - Exposes pure helpers for confidence/margin/entropy and row formatting.
- Create `run_h2_3_backbone_feature_ablation.py`
  - Merges backbone feature CSV rows into existing target prediction rows.
  - Builds rich-only and fused feature dictionaries.
  - Fits per-client, per-gas Ridge heads using calibration-validation alpha selection.
  - Reports no-QC full-set test metrics and C5 nonCO wrong-route audit.
- Create `tests/test_backbone_feature_export.py`
  - Tests probability metric calculation and row schema without loading real checkpoints.
- Create `tests/test_h2_3_backbone_feature_ablation.py`
  - Tests feature-group construction, key merge behavior, and C5 nonCO wrong-route audit.
- Create result directories by running scripts:
  - `results/f6_r25_backbone_feature_export_20260630/`
  - `results/f6_r19_backbone_feature_export_20260630_diagnostic/`
  - `results/h2_3_backbone_feature_ablation_20260630/r25/`

## Task 1: Test Backbone Feature Helpers

**Files:**
- Create: `tests/test_backbone_feature_export.py`
- Create in Step 3: `export_backbone_features.py`

- [ ] **Step 1: Write failing tests**

```python
import math

import numpy as np

from export_backbone_features import (
    build_feature_row,
    class_probability_metrics,
    feature_column_names,
)


def test_class_probability_metrics_returns_confidence_margin_entropy_and_prediction():
    probs = np.asarray([0.1, 0.7, 0.15, 0.05], dtype=np.float64)

    metrics = class_probability_metrics(probs)

    assert metrics["pred_class"] == 1
    assert metrics["confidence"] == 0.7
    assert metrics["margin"] == 0.55
    expected_entropy = -sum(float(p) * math.log(float(p)) for p in probs)
    assert abs(metrics["entropy"] - expected_entropy) < 1e-12


def test_feature_column_names_are_stable_for_cls_and_reg_features():
    names = feature_column_names(num_classes=4, cls_dim=2, reg_dim=3)

    assert names == [
        "pred_class_f6_r25",
        "prob_0",
        "prob_1",
        "prob_2",
        "prob_3",
        "confidence",
        "margin",
        "entropy",
        "cls_feat_000",
        "cls_feat_001",
        "reg_feat_000",
        "reg_feat_001",
        "reg_feat_002",
    ]


def test_build_feature_row_uses_alignment_key_and_feature_values():
    probs = np.asarray([0.1, 0.7, 0.15, 0.05], dtype=np.float64)
    cls_feat = np.asarray([1.0, 2.0], dtype=np.float64)
    reg_feat = np.asarray([3.0, 4.0, 5.0], dtype=np.float64)

    row = build_feature_row(
        client="C4",
        split="test",
        sample_index=12,
        probs=probs,
        cls_feat=cls_feat,
        reg_feat=reg_feat,
        pred_prefix="f6_r25",
    )

    assert row["client"] == "C4"
    assert row["split"] == "test"
    assert row["sample_index"] == 12
    assert row["pred_class_f6_r25"] == 1
    assert row["prob_1"] == 0.7
    assert row["cls_feat_001"] == 2.0
    assert row["reg_feat_002"] == 5.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_backbone_feature_export.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'export_backbone_features'`.

- [ ] **Step 3: Implement minimal helper functions and CLI skeleton**

Create `export_backbone_features.py` with:

```python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from gaps_flower.evaluate_checkpoint import load_checkpoint_model, make_loader, resolve_device


def class_probability_metrics(probs: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(probs, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("probability vector is empty")
    order = np.argsort(-values)
    top1 = float(values[order[0]])
    top2 = float(values[order[1]]) if values.size > 1 else 0.0
    entropy = float(-(values * np.log(np.maximum(values, 1e-12))).sum())
    return {
        "pred_class": int(order[0]),
        "confidence": top1,
        "margin": float(top1 - top2),
        "entropy": entropy,
    }


def feature_column_names(num_classes: int, cls_dim: int, reg_dim: int, pred_prefix: str = "f6_r25") -> list[str]:
    return [
        f"pred_class_{pred_prefix}",
        *[f"prob_{idx}" for idx in range(num_classes)],
        "confidence",
        "margin",
        "entropy",
        *[f"cls_feat_{idx:03d}" for idx in range(cls_dim)],
        *[f"reg_feat_{idx:03d}" for idx in range(reg_dim)],
    ]


def build_feature_row(
    *,
    client: str,
    split: str,
    sample_index: int,
    probs: np.ndarray,
    cls_feat: np.ndarray,
    reg_feat: np.ndarray,
    pred_prefix: str,
) -> dict[str, Any]:
    metrics = class_probability_metrics(probs)
    row: dict[str, Any] = {
        "client": client,
        "split": split,
        "sample_index": int(sample_index),
        f"pred_class_{pred_prefix}": metrics["pred_class"],
        "confidence": metrics["confidence"],
        "margin": metrics["margin"],
        "entropy": metrics["entropy"],
    }
    for idx, value in enumerate(np.asarray(probs, dtype=np.float64).reshape(-1)):
        row[f"prob_{idx}"] = float(value)
    for idx, value in enumerate(np.asarray(cls_feat, dtype=np.float64).reshape(-1)):
        row[f"cls_feat_{idx:03d}"] = float(value)
    for idx, value in enumerate(np.asarray(reg_feat, dtype=np.float64).reshape(-1)):
        row[f"reg_feat_{idx:03d}"] = float(value)
    return row
```

Then add CSV writer, checkpoint extraction, `run(args)`, and CLI as described in Task 2.

- [ ] **Step 4: Run tests and verify they pass**

Run: `pytest tests/test_backbone_feature_export.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_backbone_feature_export.py export_backbone_features.py
git commit -m "feat: add backbone feature export helpers"
```

## Task 2: Finish And Run P0 Feature Export

**Files:**
- Modify: `export_backbone_features.py`

- [ ] **Step 1: Add extraction implementation**

Implement these functions in `export_backbone_features.py`:

```python
def parse_clients(text: str) -> list[int]:
    return [int(item.strip().upper().replace("C", "")) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_split(
    *,
    model: torch.nn.Module,
    data_root: str | Path,
    client_ids: list[int],
    split: str,
    batch_size: int,
    device: torch.device,
    pred_prefix: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for client_id in client_ids:
            loader = make_loader(data_root, client_id, split, batch_size)
            offset = 0
            for batch in loader:
                x = batch[0].to(device)
                logits, cls_feat, reg_feat = model(x)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                cls_np = cls_feat.detach().cpu().numpy()
                reg_np = reg_feat.detach().cpu().numpy()
                for local_idx in range(probs.shape[0]):
                    rows.append(
                        build_feature_row(
                            client=f"C{client_id}",
                            split=split,
                            sample_index=offset + local_idx,
                            probs=probs[local_idx],
                            cls_feat=cls_np[local_idx],
                            reg_feat=reg_np[local_idx],
                            pred_prefix=pred_prefix,
                        )
                    )
                offset += probs.shape[0]
    return rows
```

Add `run(args)` to write `backbone_features_<split>.csv` for each requested split and a manifest containing checkpoint, round, adaptive, clients, splits, row counts, feature dimensions, and `diagnostic_only`.

- [ ] **Step 2: Run helper tests**

Run: `pytest tests/test_backbone_feature_export.py -q`

Expected: PASS.

- [ ] **Step 3: Run official r25 export**

Run:

```powershell
python export_backbone_features.py `
  --checkpoint results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/server_latest_adapted.pth `
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --clients 3,4,5 `
  --splits calibration,test `
  --pred-prefix f6_r25 `
  --output-dir results/f6_r25_backbone_feature_export_20260630
```

Expected: writes calibration/test CSVs and manifest.

- [ ] **Step 4: Run diagnostic r19 export**

Run:

```powershell
python export_backbone_features.py `
  --checkpoint results/source_target_classification_matrix_20260627/F6_C12_to_C345_fixed_da_strong_r25/server_round_019_adapted.pth `
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --clients 3,4,5 `
  --splits calibration,test `
  --pred-prefix f6_r19 `
  --diagnostic-only `
  --output-dir results/f6_r19_backbone_feature_export_20260630_diagnostic
```

Expected: manifest contains `"diagnostic_only": true`.

- [ ] **Step 5: Commit Task 2**

```bash
git add export_backbone_features.py results/f6_r25_backbone_feature_export_20260630 results/f6_r19_backbone_feature_export_20260630_diagnostic
git commit -m "feat: export F6 backbone features"
```

## Task 3: Test H2.3 Feature Ablation Helpers

**Files:**
- Create: `tests/test_h2_3_backbone_feature_ablation.py`
- Create in Step 3: `run_h2_3_backbone_feature_ablation.py`

- [ ] **Step 1: Write failing tests**

```python
from run_h2_3_backbone_feature_ablation import (
    build_feature_groups,
    c5_nonco_wrong_route_audit,
    merge_backbone_features,
)


def test_merge_backbone_features_matches_client_split_sample_index():
    rows = [{"client": "C3", "split": "test", "sample_index": "7", "feature_dict": {"rich": 1.0}}]
    features = [{"client": "C3", "split": "test", "sample_index": "7", "confidence": "0.8", "reg_feat_000": "1.5"}]

    merged = merge_backbone_features(rows, features)

    assert merged[0]["backbone_feature_dict"]["confidence"] == 0.8
    assert merged[0]["backbone_feature_dict"]["reg_feat_000"] == 1.5


def test_build_feature_groups_separates_embedding_and_b0_priors():
    row = {
        "feature_dict": {"rich": 1.0},
        "backbone_feature_dict": {
            "confidence": 0.8,
            "prob_0": 0.1,
            "cls_feat_000": 2.0,
            "reg_feat_000": 3.0,
        },
        "final_ppm": "42.0",
        "base_r3ak16_raw_ppm": "41.0",
        "routed_pred_ppm": "40.0",
    }

    groups = build_feature_groups(row)

    assert groups["A0_rich_only"] == {"rich": 1.0}
    assert "reg_feat_000" in groups["A3_rich_plus_reg_feat"]
    assert "final_ppm" in groups["A4_rich_plus_b0"]
    assert "routed_pred_ppm" in groups["A5_rich_plus_source_priors"]
    assert "reg_feat_000" in groups["A7_rich_plus_all_priors"]
    assert "final_ppm" in groups["A7_rich_plus_all_priors"]


def test_c5_nonco_wrong_route_audit_counts_nonco_as_co_routes():
    rows = [
        {"client": "C5", "true_class": "0", "pred_class": "1"},
        {"client": "C5", "true_class": "2", "pred_class": "1"},
        {"client": "C5", "true_class": "1", "pred_class": "1"},
        {"client": "C4", "true_class": "0", "pred_class": "1"},
    ]

    audit = c5_nonco_wrong_route_audit(rows)

    assert audit["C5_nonCO_N"] == 2
    assert audit["C5_nonCO_pred_CO_N"] == 2
    assert audit["C5_nonCO_pred_CO_rate"] == 1.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/test_h2_3_backbone_feature_ablation.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'run_h2_3_backbone_feature_ablation'`.

- [ ] **Step 3: Implement minimal helper functions**

Create `run_h2_3_backbone_feature_ablation.py` with helper functions:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from run_formal_target_ridge_auto_v2_eval import fit_client_models, refit_full_calibration
from run_regression_head_ablation import (
    CO_CLASS,
    add_target_features,
    apply_client_models,
    client_name,
    client_num,
    fnum,
    inum,
    read_csv,
    summarize,
    write_csv,
)
```

Add `merge_backbone_features`, `build_feature_groups`, and `c5_nonco_wrong_route_audit` matching the tests.

- [ ] **Step 4: Run helper tests**

Run: `pytest tests/test_h2_3_backbone_feature_ablation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add tests/test_h2_3_backbone_feature_ablation.py run_h2_3_backbone_feature_ablation.py
git commit -m "feat: add backbone ablation helpers"
```

## Task 4: Finish And Run P1 Feature Ablation

**Files:**
- Modify: `run_h2_3_backbone_feature_ablation.py`

- [ ] **Step 1: Add ablation execution**

Implement `run(args)` to:

1. Read target predictions from official r25 target layer predictions.
2. Add rich target features with `add_target_features`.
3. Merge calibration and test backbone feature CSVs.
4. Build each feature group by replacing `row["feature_dict"]` with the selected group.
5. Fit per-client, per-gas Ridge heads using `fit_client_models`.
6. Refit on full calibration using selected alphas.
7. Apply to test rows with `route_class=pred_class`.
8. Summarize each mode with existing `summarize`.
9. Write:
   - `feature_ablation_predictions.csv`
   - `feature_ablation_summary.csv`
   - `feature_ablation_fit_audit.csv`
   - `feature_ablation_wrong_route_audit.csv`
   - `feature_ablation_report.md`
   - `manifest.json`

- [ ] **Step 2: Run tests**

Run:

```powershell
pytest tests/test_backbone_feature_export.py tests/test_h2_3_backbone_feature_ablation.py -q
```

Expected: PASS.

- [ ] **Step 3: Run official r25 feature ablation**

Run:

```powershell
python run_h2_3_backbone_feature_ablation.py `
  --data-root dataset/client_data_c12src_c345tgt_2080_timeaware_60_170_window_fullgrid `
  --target-predictions results/f6_c12_c345_strong_r25_r3ak16_auto_v2_eval/ppm_layer_co_audit/target_layer_predictions.csv `
  --backbone-calibration results/f6_r25_backbone_feature_export_20260630/backbone_features_calibration.csv `
  --backbone-test results/f6_r25_backbone_feature_export_20260630/backbone_features_test.csv `
  --target-clients 3,4,5 `
  --output-dir results/h2_3_backbone_feature_ablation_20260630/r25
```

Expected: report compares A0-A7 and includes C5 nonCO wrong-route audit.

- [ ] **Step 4: Analyze report**

Read `results/h2_3_backbone_feature_ablation_20260630/r25/feature_ablation_report.md`.

Classify the result:

- Backbone-positive: A2/A3/A6/A7 beat A0 without nonCO_ALL degradation > 1.0 RMSE.
- Prior-positive: only A4/A5/A7 beat A0.
- Negative: no fused feature group improves A0 materially.

- [ ] **Step 5: Commit Task 4**

```bash
git add run_h2_3_backbone_feature_ablation.py results/h2_3_backbone_feature_ablation_20260630/r25
git commit -m "feat: run H2.3 backbone feature ablation"
```

## Task 5: Verification And Feedback

**Files:**
- No required code edits unless Task 4 exposes a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
pytest tests/test_backbone_feature_export.py tests/test_h2_3_backbone_feature_ablation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing contract smoke tests**

Run:

```powershell
pytest tests/test_flower_classification_contract.py tests/test_regression_mainline_integrity.py -q
```

Expected: PASS.

- [ ] **Step 3: Report results**

Return:

- Official r25 feature export row counts.
- Best A0-A7 mode by ALL RMSE and macro-client NRMSE.
- Whether the result is backbone-positive, prior-positive, or negative.
- C5 nonCO wrong-route audit.
- Recommendation for P2:
  - If backbone-positive or prior-positive, implement H2.3+ fusion profile next.
  - If negative, keep H2.3 current and defer to regression-aware encoder work.

- [ ] **Step 4: Commit any final report or docs adjustment**

Only commit if a report summary file is added or code changed after Task 4.
