from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

import gaps_deploy.package_contract as package_contract
import gaps_deploy.build_package as build_package_module
from gaps_deploy.build_package import build_package
from gaps_deploy.build_per_client_packages import _build_package_command
from gaps_deploy.deploy_config import DeployConfig
from gaps_deploy.inference import DeployPredictor
from gaps_deploy.validate_deployment_packages import validate_packages
from scripts.build_final_deployment_package import copy_runtime_source


def test_empty_package_never_constructs_random_models(tmp_path) -> None:
    with pytest.raises(ValueError, match="deploy_config.json"):
        DeployPredictor.from_package(str(tmp_path))


def test_empty_config_never_constructs_random_models() -> None:
    with pytest.raises(ValueError, match="classifier_checkpoint"):
        DeployPredictor.from_config(DeployConfig())


@pytest.mark.parametrize("damage", ["missing", "unexpected", "shape"])
def test_state_dict_contract_rejects_incompatible_weights(damage: str) -> None:
    model = torch.nn.Linear(2, 1)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    if damage == "missing":
        state.pop("bias")
    elif damage == "unexpected":
        state["extra"] = torch.zeros(1)
    else:
        state["weight"] = torch.zeros(1, 3)

    with pytest.raises(ValueError, match="fixture.pth"):
        package_contract.load_state_dict_strict(model, state, Path("fixture.pth"))


def test_state_dict_contract_accepts_an_exact_state() -> None:
    model = torch.nn.Linear(2, 1)
    state = {key: value.clone() for key, value in model.state_dict().items()}

    package_contract.load_state_dict_strict(model, state, Path("fixture.pth"))


def test_checkpoint_loader_extracts_a_real_state_dict(tmp_path) -> None:
    path = tmp_path / "model.pth"
    torch.save({"model_state": torch.nn.Linear(2, 1).state_dict()}, path)

    checkpoint, state = package_contract.load_checkpoint_state(path)

    assert "model_state" in checkpoint
    assert set(state) == {"weight", "bias"}


@pytest.mark.parametrize("payload", [{}, {"model_state": {}}, {"model_state": {"weight": 1.0}}])
def test_checkpoint_loader_rejects_malformed_state(tmp_path, payload) -> None:
    path = tmp_path / "bad.pth"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="bad.pth"):
        package_contract.load_checkpoint_state(path)


def _none_routing() -> dict:
    return {
        "selected_modes": {str(class_id): "none" for class_id in range(4)},
        "affine_params": {},
        "phase_affine_params": {},
    }


def test_routing_contract_normalizes_complete_class_keys() -> None:
    normalized = package_contract.normalize_and_validate_routing_config(
        _none_routing(), num_classes=4
    )

    assert normalized["selected_modes"] == {
        0: "none",
        1: "none",
        2: "none",
        3: "none",
    }


@pytest.mark.parametrize("damage", ["missing", "extra", "collision", "unknown", "missing_params"])
def test_routing_contract_rejects_incomplete_or_ambiguous_routes(damage: str) -> None:
    routing = _none_routing()
    if damage == "missing":
        routing["selected_modes"].pop("3")
    elif damage == "extra":
        routing["selected_modes"]["4"] = "none"
    elif damage == "collision":
        routing["selected_modes"][0] = "none"
    elif damage == "unknown":
        routing["selected_modes"]["0"] = "raw_fallback"
    else:
        routing["selected_modes"]["0"] = "affine_only"

    with pytest.raises(ValueError, match="routing_config"):
        package_contract.normalize_and_validate_routing_config(routing, num_classes=4)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _model_config() -> dict:
    return {
        "num_classes": 4,
        "num_sensors": 8,
        "feat_dim": 64,
        "encoder_type": "tcn",
        "transformer_d_model": 48,
        "transformer_nhead": 4,
        "transformer_num_layers": 2,
        "transformer_ff_dim": 96,
        "reg_head_depth": 3,
        "reg_output_mode": "sigmoid",
        "reg_window_stats": False,
        "reg_window_stats_mode": "global",
        "reg_window_stats_dim": 8,
        "reg_response_branch": "none",
        "reg_tcn_adapter": False,
        "reg_use_shared_trunk": False,
        "use_reg_ratio_branch": False,
    }


def _minimal_package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    config = DeployConfig()
    config.classifier_checkpoint = "models/classification_model.pth"
    config.regression_checkpoint = "models/regression_model.pth"
    config.routing_config_path = "calibration/routing_config.json"
    config.qc_policy_path = "qc/selected_policy.json"
    _write_json(package / "config/deploy_config.json", config.to_dict())
    _write_json(package / "models/model_config.json", _model_config())
    _write_json(package / "calibration/routing_config.json", _none_routing())
    _write_json(
        package / "qc/selected_policy.json",
        {
            "policies": [
                {
                    "policy_name": "classifier_only",
                    "scores": ["classifier_uncertainty"],
                    "thresholds": {"classifier_uncertainty": 0.5},
                    "low_ratio": 0.9,
                    "high_ratio": 1.1,
                    "group": "ALL",
                }
            ]
        },
    )
    model = torch.nn.Linear(2, 1)
    torch.save({"model_state": model.state_dict(), "round": 25}, package / "models/classification_model.pth")
    torch.save({"model_state": model.state_dict()}, package / "models/regression_model.pth")
    return package


@pytest.fixture
def tiny_model_factories(monkeypatch):
    monkeypatch.setattr(
        DeployPredictor,
        "_create_classifier_model",
        staticmethod(lambda model_config, deploy_config: torch.nn.Linear(2, 1)),
    )
    monkeypatch.setattr(
        DeployPredictor,
        "_create_regression_model",
        staticmethod(lambda model_config, deploy_config: torch.nn.Linear(2, 1)),
    )


def test_complete_package_loads_exact_checkpoints(tmp_path, tiny_model_factories) -> None:
    predictor = DeployPredictor.from_package(str(_minimal_package(tmp_path)))

    assert predictor.model_version == "25"


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/model_config.json",
        "models/classification_model.pth",
        "models/regression_model.pth",
        "calibration/routing_config.json",
        "qc/selected_policy.json",
    ],
)
def test_package_requires_every_production_asset(
    tmp_path, tiny_model_factories, relative_path: str
) -> None:
    package = _minimal_package(tmp_path)
    (package / relative_path).unlink()

    with pytest.raises(ValueError, match=Path(relative_path).name):
        DeployPredictor.from_package(str(package))


@pytest.mark.parametrize("checkpoint_name", ["classification_model.pth", "regression_model.pth"])
def test_package_rejects_partial_checkpoint_state(
    tmp_path, tiny_model_factories, checkpoint_name: str
) -> None:
    package = _minimal_package(tmp_path)
    torch.save(
        {"model_state": {"bias": torch.zeros(1)}},
        package / "models" / checkpoint_name,
    )

    with pytest.raises(ValueError, match=checkpoint_name):
        DeployPredictor.from_package(str(package))


@pytest.mark.parametrize("mode", ["full", "specialist_full"])
def test_package_requires_model_selected_by_routing(
    tmp_path, tiny_model_factories, mode: str
) -> None:
    package = _minimal_package(tmp_path)
    routing = _none_routing()
    routing["selected_modes"]["0"] = mode
    _write_json(package / "calibration/routing_config.json", routing)

    with pytest.raises(ValueError, match="class 0|full_model.pth"):
        DeployPredictor.from_package(str(package))


def _minimal_direct_config(tmp_path: Path) -> DeployConfig:
    package = _minimal_package(tmp_path)
    config = DeployConfig()
    config.model_config = _model_config()
    config.classifier_checkpoint = str(package / "models/classification_model.pth")
    config.regression_checkpoint = str(package / "models/regression_model.pth")
    config.routing_config_path = str(package / "calibration/routing_config.json")
    config.qc_policy_path = str(package / "qc/selected_policy.json")
    return config


def test_from_config_rejects_partial_checkpoint_state(
    tmp_path, tiny_model_factories
) -> None:
    config = _minimal_direct_config(tmp_path)
    torch.save(
        {"model_state": {"bias": torch.zeros(1)}},
        config.classifier_checkpoint,
    )

    with pytest.raises(ValueError, match="classification_model.pth"):
        DeployPredictor.from_config(config)


def test_from_config_requires_routing_and_qc_assets(
    tmp_path, tiny_model_factories
) -> None:
    config = _minimal_direct_config(tmp_path)
    config.routing_config_path = ""

    with pytest.raises(ValueError, match="routing_config_path"):
        DeployPredictor.from_config(config)


def test_package_requires_explicit_core_model_schema(
    tmp_path, tiny_model_factories
) -> None:
    package = _minimal_package(tmp_path)
    model_config = _model_config()
    model_config.pop("num_sensors")
    _write_json(package / "models/model_config.json", model_config)

    with pytest.raises(ValueError, match="num_sensors"):
        DeployPredictor.from_package(str(package))


def test_package_rejects_checkpoint_model_config_disagreement(
    tmp_path, tiny_model_factories
) -> None:
    package = _minimal_package(tmp_path)
    model = torch.nn.Linear(2, 1)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"reg_head_depth": 9},
        },
        package / "models/regression_model.pth",
    )

    with pytest.raises(ValueError, match="reg_head_depth"):
        DeployPredictor.from_package(str(package))


def test_package_rejects_specialist_model_config_disagreement(
    tmp_path, tiny_model_factories
) -> None:
    package = _minimal_package(tmp_path)
    routing = _none_routing()
    routing["selected_modes"]["0"] = "specialist"
    _write_json(package / "calibration/routing_config.json", routing)
    model = torch.nn.Linear(2, 1)
    specialist = package / "models/specialists/class_0.pth"
    specialist.parent.mkdir(parents=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {"reg_head_depth": 9},
        },
        specialist,
    )

    with pytest.raises(ValueError, match="reg_head_depth"):
        DeployPredictor.from_package(str(package))


def _builder_sources(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "sources"
    source.mkdir(parents=True)
    model = torch.nn.Linear(2, 1)
    classifier = source / "classifier.pth"
    regression = source / "regression.pth"
    torch.save({"model_state": model.state_dict()}, classifier)
    torch.save(
        {"model_state": model.state_dict(), "model_config": {"reg_head_depth": 3}},
        regression,
    )
    model_config = source / "model_config.json"
    _write_json(model_config, _model_config())
    calibration = source / "calibration"
    _write_json(calibration / "routing_config.json", _none_routing())
    qc_policy = source / "selected_policy.json"
    _write_json(
        qc_policy,
        {
            "policies": [
                {
                    "policy_name": "classifier_only",
                    "scores": ["classifier_uncertainty"],
                    "thresholds": {"classifier_uncertainty": 0.5},
                    "low_ratio": 0.9,
                    "high_ratio": 1.1,
                    "group": "ALL",
                }
            ]
        },
    )
    return {
        "classifier": classifier,
        "regression": regression,
        "model_config": model_config,
        "calibration": calibration,
        "qc_policy": qc_policy,
    }


def test_builder_requires_explicit_model_config(tmp_path) -> None:
    src = _builder_sources(tmp_path)

    with pytest.raises(ValueError, match="model_config"):
        build_package(
            output_dir=str(tmp_path / "package"),
            classifier_ckpt=str(src["classifier"]),
            regression_ckpt=str(src["regression"]),
            calibration_dir=str(src["calibration"]),
            qc_policy=str(src["qc_policy"]),
        )


def test_builder_requires_explicit_qc_policy(tmp_path) -> None:
    src = _builder_sources(tmp_path)

    with pytest.raises(ValueError, match="qc_policy"):
        build_package(
            output_dir=str(tmp_path / "package"),
            classifier_ckpt=str(src["classifier"]),
            regression_ckpt=str(src["regression"]),
            model_config_path=str(src["model_config"]),
            calibration_dir=str(src["calibration"]),
            qc_policy="",
        )


def _build_from_sources(tmp_path: Path, src: dict[str, Path], **overrides) -> Path:
    kwargs = {
        "output_dir": str(tmp_path / "package"),
        "classifier_ckpt": str(src["classifier"]),
        "regression_ckpt": str(src["regression"]),
        "model_config_path": str(src["model_config"]),
        "calibration_dir": str(src["calibration"]),
        "qc_policy": str(src["qc_policy"]),
    }
    kwargs.update(overrides)
    return build_package(**kwargs)


def test_builder_copies_explicit_model_config_exactly(tmp_path) -> None:
    src = _builder_sources(tmp_path)

    package = _build_from_sources(tmp_path, src)

    assert json.loads((package / "models/model_config.json").read_text(encoding="utf-8")) == _model_config()


def test_builder_rejects_legacy_override_disagreement(tmp_path) -> None:
    src = _builder_sources(tmp_path)

    with pytest.raises(ValueError, match="reg_head_depth"):
        _build_from_sources(tmp_path, src, reg_head_depth=5)


def test_builder_rejects_checkpoint_model_config_disagreement(tmp_path) -> None:
    src = _builder_sources(tmp_path)
    model = torch.nn.Linear(2, 1)
    torch.save(
        {"model_state": model.state_dict(), "model_config": {"reg_head_depth": 9}},
        src["regression"],
    )

    with pytest.raises(ValueError, match="reg_head_depth"):
        _build_from_sources(tmp_path, src)


def test_builder_requires_routing_config(tmp_path) -> None:
    src = _builder_sources(tmp_path)
    (src["calibration"] / "routing_config.json").unlink()

    with pytest.raises(ValueError, match="routing_config.json"):
        _build_from_sources(tmp_path, src)


def test_builder_requires_full_checkpoint_when_routing_selects_full(tmp_path) -> None:
    src = _builder_sources(tmp_path)
    routing = _none_routing()
    routing["selected_modes"]["0"] = "full"
    _write_json(src["calibration"] / "routing_config.json", routing)

    with pytest.raises(ValueError, match="full_model"):
        _build_from_sources(tmp_path, src)


def test_builder_requires_response_references_for_response_qc(tmp_path) -> None:
    src = _builder_sources(tmp_path)
    _write_json(
        src["qc_policy"],
        {
            "policies": [
                {
                    "policy_name": "response_qc",
                    "scores": ["response_signature_norm"],
                    "thresholds": {"response_signature_norm": 1.0},
                    "low_ratio": 0.9,
                    "high_ratio": 1.1,
                    "group": "ALL",
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="calibration_stats.json"):
        _build_from_sources(tmp_path, src)


def test_builder_cli_requires_explicit_contract_assets(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_package",
            "--output-dir",
            "out",
            "--classifier-ckpt",
            "classifier.pth",
            "--regression-ckpt",
            "regression.pth",
            "--calibration-dir",
            "calibration",
            "--qc-policy",
            "policy.json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        build_package_module.main()

    assert exc_info.value.code == 2


def _validate_single_package(package: Path) -> dict:
    return validate_packages(
        deploy_package=str(package),
        client_packages=[],
        clients=["3"],
        require_distinct_packages=False,
        require_response_refs=False,
        min_response_ref_classes=4,
        expected_reg_head_depth=0,
    )


def test_validator_loads_complete_package_strictly(
    tmp_path, tiny_model_factories
) -> None:
    report = _validate_single_package(_minimal_package(tmp_path))

    assert report["status"] == "pass"


def test_validator_rejects_malformed_qc_policy(
    tmp_path, tiny_model_factories
) -> None:
    package = _minimal_package(tmp_path)
    (package / "qc/selected_policy.json").write_text("{", encoding="utf-8")

    report = _validate_single_package(package)

    assert report["status"] == "fail"
    assert any("QC" in error or "policy" in error for error in report["packages"][0]["errors"])


def test_validator_rejects_partial_checkpoint_state(
    tmp_path, tiny_model_factories
) -> None:
    package = _minimal_package(tmp_path)
    torch.save(
        {"model_state": {"bias": torch.zeros(1)}},
        package / "models/classification_model.pth",
    )

    report = _validate_single_package(package)

    assert report["status"] == "fail"
    assert any("classification_model.pth" in error for error in report["packages"][0]["errors"])


def test_per_client_builder_propagates_required_contract_assets(tmp_path) -> None:
    args = SimpleNamespace(
        classifier_ckpt="classifier.pth",
        regression_ckpt="regression.pth",
        model_config="model_config.json",
        qc_policy="selected_policy.json",
        model_version="v1",
        reg_head_depth=3,
        reg_output_mode="",
        reg_window_stats=False,
        reg_window_stats_mode="global",
        reg_window_stats_dim=8,
        reg_response_branch="",
        reg_dct_k=None,
        reg_dct_gamma_init=None,
        reg_dct_dropout=None,
        reg_msconv_channels=None,
        reg_msconv_kernels="",
        reg_msconv_gamma_init=None,
        reg_msconv_dropout=None,
        reg_tcn_adapter=False,
        reg_tcn_adapter_kernel=None,
        reg_tcn_adapter_gamma_init=None,
        reg_tcn_adapter_dropout=None,
        reg_use_shared_trunk=False,
        reg_shared_trunk_dim=None,
        reg_gas_emb_dim=None,
        reg_residual_head_depth=None,
        use_reg_ratio_branch=False,
        reg_ratio_gamma_init=None,
        reg_ratio_dropout=None,
    )

    command = _build_package_command(
        args,
        "C3",
        tmp_path / "calibration",
        tmp_path / "package",
    )

    assert command[command.index("--model-config") + 1] == "model_config.json"
    assert command[command.index("--qc-policy") + 1] == "selected_policy.json"


def test_final_bundle_includes_strict_package_contract(tmp_path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    copy_runtime_source(repo_root, tmp_path / "bundle")

    assert (
        tmp_path / "bundle/runtime_src/gaps_deploy/package_contract.py"
    ).is_file()
