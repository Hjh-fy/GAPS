from pathlib import Path

import pytest

from scripts.summarize_iotj_b5_server_da_budget_ablation import (
    _prediction_map,
)
from scripts.summarize_iotj_b5_rejected_observability_attempt import (
    _finite,
)


def test_prediction_map_rejects_non_1360_rows(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text(
        "row_key,true_class,pred_class\nrow-1,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="invalid prediction rows"):
        _prediction_map(path)


def test_noncanonical_receipt_finite_check_rejects_nan() -> None:
    assert _finite({"metrics": [0.1, 0.2]})
    assert not _finite({"metrics": [0.1, float("nan")]})
