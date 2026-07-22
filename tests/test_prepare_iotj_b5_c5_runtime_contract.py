from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(path: Path) -> Path:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps({"schema_version": "iotj.b5_c5_deployment_bundle.v1", "status": "ready"}),
        encoding="utf-8",
    )
    return path


def test_prepare_contract_binds_model_inputs_and_both_hc_references(tmp_path: Path) -> None:
    from scripts.prepare_iotj_b5_c5_runtime_contract import prepare_runtime_contract

    bundle = _write_bundle(tmp_path / "bundle")
    features = tmp_path / "test_features.npy"
    np.save(features, np.zeros((1360, 100, 8), dtype=np.float32))
    metadata = tmp_path / "test_experiment_info.json"
    metadata.write_text(json.dumps([{"row": index} for index in range(1360)]), encoding="utf-8")
    hc95, hc90 = tmp_path / "hc95.csv", tmp_path / "hc90.csv"
    hc95.write_text("sample_index\n" + "\n".join(map(str, range(1360))) + "\n", encoding="utf-8")
    hc90.write_bytes(hc95.read_bytes())

    output = prepare_runtime_contract(
        bundle_dir=bundle,
        classifier_model={"architecture": "FedGasBaseModel", "encoder_type": "tcn", "num_classes": 4, "num_sensors": 8, "feat_dim": 64, "tcn_norm": "instance", "use_cls_proj": False},
        input_features=features,
        input_metadata=metadata,
        hc95_reference=hc95,
        hc90_reference=hc90,
        output_dir=tmp_path / "contract",
    )

    payload = json.loads((output / "runtime_contract.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "iotj.c5_h8_runtime_contract.v1"
    assert payload["inputs"]["features"]["sha256"] == _sha256(features)
    assert set(payload["references"]) == {"HC95", "HC90"}


def test_prepare_contract_refuses_to_overwrite_or_bind_wrong_shape(tmp_path: Path) -> None:
    from scripts.prepare_iotj_b5_c5_runtime_contract import prepare_runtime_contract

    bundle = _write_bundle(tmp_path / "bundle")
    features = tmp_path / "bad.npy"
    np.save(features, np.zeros((2, 100, 8), dtype=np.float32))
    metadata = tmp_path / "meta.json"
    metadata.write_text("[]", encoding="utf-8")
    ref = tmp_path / "ref.csv"
    ref.write_text("sample_index\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1360"):
        prepare_runtime_contract(bundle_dir=bundle, classifier_model={}, input_features=features, input_metadata=metadata, hc95_reference=ref, hc90_reference=ref, output_dir=tmp_path / "out")
