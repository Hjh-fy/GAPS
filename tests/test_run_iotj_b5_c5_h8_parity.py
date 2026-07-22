from pathlib import Path

import pytest


def test_runner_refuses_nonempty_output_before_loading_assets(tmp_path: Path) -> None:
    from scripts.run_iotj_b5_c5_h8_parity import run_c5_h8_parity

    output = tmp_path / "existing"
    output.mkdir()
    (output / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_c5_h8_parity(contract_path=tmp_path / "missing.json", row_map_path=tmp_path / "missing-map.json", output_dir=output)


def test_row_map_rejects_wrong_contract_hash(tmp_path: Path) -> None:
    import json
    from scripts.run_iotj_b5_c5_h8_parity import _load_row_map

    contract = tmp_path / "contract.json"
    contract.write_text("{}", encoding="utf-8")
    row_map = tmp_path / "row-map.json"
    row_map.write_text(json.dumps({"schema_version": "iotj.c5_h8_row_map.v1", "status": "ready", "row_count": 1360, "contract_sha256": "0" * 64, "rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="contract hash"):
        _load_row_map(row_map, contract)
