from collections import OrderedDict

import numpy as np
import pytest
import torch
import flwr as fl
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

import gaps_flower.client_app as flower_client_module
from gaps_flower.client_app import GapsFlowerClient
from gaps_flower.strategy import CheckpointFedAvg, GapsStrategy
from gaps_flower.task import create_model, evaluate, get_parameters, make_config, set_parameters
from scripts.generate_iotj_classification_ablation_commands import (
    SPECS,
    _write_command_files,
    build_run_manifest,
)


def test_flower_config_is_simplified_classifier_only() -> None:
    cfg = make_config(device="cpu", local_epochs=1, batch_size=4)

    assert cfg.USE_REG_LOSS is False
    assert cfg.USE_ALIGN is False
    assert cfg.USE_REPLAY_DISTILL is False
    assert cfg.USE_SERVER_OPT is False
    assert cfg.USE_MMD_ALIGNMENT is False
    assert cfg.USE_DEEP_CORAL is False
    assert cfg.USE_ADVERSARIAL_DOMAIN is False
    assert cfg.USE_SENSOR_AUG is False

    model = create_model(cfg)
    state_keys = set(model.state_dict())

    assert not any(key.startswith("reg_heads.") for key in state_keys)
    assert not any(key.startswith("reg_response_adapter.") for key in state_keys)
    assert not any(key.startswith("reg_ratio_adapter.") for key in state_keys)
    assert not any(key.startswith("reg_shared_trunk.") for key in state_keys)


@pytest.mark.parametrize(
    ("profile", "use_align", "use_replay", "use_decouple"),
    [
        ("ce_only", False, False, False),
        ("align_only", True, False, False),
        ("align_replay", True, True, False),
        ("proto_only", True, False, True),
        ("replay_only", False, True, False),
        ("proto_replay", True, True, True),
    ],
)
def test_classification_ablation_profiles_have_exact_switches(
    profile: str,
    use_align: bool,
    use_replay: bool,
    use_decouple: bool,
) -> None:
    cfg = make_config(profile=profile, seed=47)

    assert cfg.SEED == 47
    assert cfg.USE_REG_LOSS is False
    assert cfg.USE_ALIGN is use_align
    assert cfg.USE_CONTRASTIVE_ALIGN is use_align
    assert cfg.USE_REPLAY_DISTILL is use_replay
    assert cfg.USE_PROTO_DECOUPLING is use_decouple
    assert cfg.UPLOAD_PROTO_STATS is (use_align or use_decouple)


@pytest.mark.parametrize("alias", ["smoke", "strong_cls", "gaps_cls"])
def test_legacy_classification_profiles_remain_supported(alias: str) -> None:
    cfg = make_config(profile=alias)
    expected_enabled = alias != "smoke"

    assert cfg.USE_ALIGN is expected_enabled
    assert cfg.USE_REPLAY_DISTILL is expected_enabled


def test_flower_parameter_order_round_trip_is_stable() -> None:
    cfg = make_config(device="cpu", local_epochs=1, batch_size=4)
    source = create_model(cfg)
    target = create_model(cfg)

    arrays, keys = get_parameters(source)
    assert keys == list(source.state_dict().keys())
    assert len(arrays) == len(keys)

    set_parameters(target, arrays, keys)
    target_arrays, target_keys = get_parameters(target)

    assert target_keys == keys
    for left, right in zip(arrays, target_arrays):
        np.testing.assert_array_equal(left, right)


def test_flower_parameter_round_trip_preserves_tensor_dtypes() -> None:
    cfg = make_config(device="cpu", local_epochs=1, batch_size=4)
    model = create_model(cfg)
    arrays, keys = get_parameters(model)

    before = OrderedDict((key, tensor.dtype) for key, tensor in model.state_dict().items())
    set_parameters(model, arrays, keys)
    after = OrderedDict((key, tensor.dtype) for key, tensor in model.state_dict().items())

    assert after == before
    with torch.no_grad():
        x = torch.randn(2, cfg.SEQ_LEN, cfg.INPUT_DIM)
        logits, _cls_feat, reg_feat = model(x)
    assert logits.shape == (2, cfg.NUM_CLASSES)
    assert reg_feat.shape[0] == 2


def test_checkpoint_fedavg_uses_flower_configure_fit() -> None:
    assert CheckpointFedAvg.configure_fit is fl.server.strategy.FedAvg.configure_fit


def test_domain_adapted_arrays_can_be_returned_as_next_global_parameters(tmp_path) -> None:
    cfg = make_config(device="cpu", local_epochs=1, batch_size=4)
    model = create_model(cfg)
    arrays, keys = get_parameters(model)
    adapted_arrays = [array.copy() for array in arrays]
    adapted_arrays[0] = adapted_arrays[0] + np.ones_like(adapted_arrays[0], dtype=adapted_arrays[0].dtype)

    strategy = GapsStrategy(
        parameter_keys=keys,
        reference_state=model.state_dict(),
        output_dir=str(tmp_path),
        run_name="test_da_return",
        use_selective_agg=False,
        use_proto_mmd=False,
        use_domain_adapt=True,
        domain_adapt_warmup=0,
        use_adapted_as_global=True,
    )

    def fake_run_domain_adapt(server_round, aggregated_state, plain_arrays, results, weights):
        return "adapted.pth", {"checkpoint_changed_tensors": 1}, adapted_arrays

    strategy._run_domain_adapt = fake_run_domain_adapt
    fit_res = FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(arrays),
        num_examples=10,
        metrics={"client_id": 1},
    )

    returned_parameters, _metrics = strategy.aggregate_fit(
        server_round=1,
        results=[(None, fit_res)],
        failures=[],
    )

    returned_arrays = parameters_to_ndarrays(returned_parameters)
    np.testing.assert_array_equal(returned_arrays[0], adapted_arrays[0])
    assert not np.array_equal(returned_arrays[0], arrays[0])


@pytest.mark.parametrize("group_id", sorted(SPECS))
def test_ablation_manifests_freeze_c12_to_c5_protocol(tmp_path, group_id: str) -> None:
    data_root = tmp_path / "dataset" / "client_data_c1234src_c5tgt_2080_timeaware_60_170_window_fullgrid"
    data_root.mkdir(parents=True)
    (data_root / "split_info.json").write_text("{}\n", encoding="utf-8")
    (data_root / "norm_stats.npz").write_bytes(b"norm")

    manifest = build_run_manifest(
        group_id,
        42,
        repo_root=tmp_path,
        results_root="results/iotj_classification_ablation_20260711",
    )

    assert manifest["protocol"]["source_clients"] == [1, 2]
    assert manifest["protocol"]["target_clients"] == [5]
    assert manifest["protocol"]["training_seed"] == 42
    assert manifest["training"]["rounds"] == 25
    assert manifest["training"]["local_epochs"] == 5
    assert manifest["training"]["batch_size"] == 32
    assert manifest["training"]["client_lr"] == 5e-4
    assert manifest["server_adaptation"]["steps"] == 100
    assert manifest["server_adaptation"]["lr"] == 5e-4
    expected_target_ce = 1.0 if group_id == "A0T" else 0.0
    assert manifest["server_adaptation"]["lambda_target_ce"] == expected_target_ce
    assert any("client_5" in arg for arg in manifest["commands"]["server_ecs"])
    assert all("client_3" not in str(command) and "client_4" not in str(command) for command in manifest["commands"].values())
    assert manifest["commands"]["client_c1_pi"][-1] == "42"
    assert "cpu" in manifest["commands"]["client_c1_pi"]
    assert "cpu" in manifest["commands"]["client_c2_pc"]


def test_primary_ablation_factors_are_causally_separated(tmp_path) -> None:
    manifests = {
        group: build_run_manifest(group, 42, repo_root=tmp_path, results_root="results/test")
        for group in ("A0", "A0T", "A2", "A3", "A4", "A4S", "A5", "A6", "A7")
    }

    assert all(
        manifests[group]["training"]["use_selective_agg"] is False
        for group in ("A0", "A2", "A3", "A4")
    )
    assert all(
        manifests[group]["training"]["use_selective_agg"] is True
        for group in ("A4S", "A5", "A6", "A7")
    )
    assert manifests["A2"]["causal_factors"]["prototype_alignment"] is True
    assert manifests["A2"]["causal_factors"]["device_residual_statistics"] is False
    assert manifests["A2"]["causal_factors"]["replay_distillation"] is False
    assert manifests["A3"]["causal_factors"]["prototype_alignment"] is False
    assert manifests["A3"]["causal_factors"]["replay_distillation"] is True
    assert manifests["A5"]["causal_factors"]["device_residual_statistics"] is False
    assert manifests["A6"]["causal_factors"]["device_residual_statistics"] is True
    assert manifests["A0T"]["causal_factors"]["target_supervised_ce"] is True
    assert all(
        manifests[group]["causal_factors"]["target_supervised_ce"] is False
        for group in ("A0", "A2", "A3", "A4", "A4S", "A5", "A6", "A7")
    )
    assert all(
        manifest["training"]["use_proto_mmd_diagnostics"] is False
        for manifest in manifests.values()
    )


def test_leave_one_group_out_specs_remove_only_declared_da_group(tmp_path) -> None:
    manifests = {
        group: build_run_manifest(group, 42, repo_root=tmp_path, results_root="results/test")
        for group in ("A7-noCORAL", "A7-noMMD", "A7-noADV", "A7-noSemantic", "A7-noStage")
    }

    assert manifests["A7-noCORAL"]["server_adaptation"]["use_coral"] is False
    no_mmd = manifests["A7-noMMD"]["server_adaptation"]
    assert no_mmd["use_mmd"] is False
    assert all(no_mmd[key] == 0.0 for key in (
        "lambda_global_mmd", "lambda_class_mmd", "lambda_proto_mmd", "lambda_stage_mmd"
    ))
    assert manifests["A7-noADV"]["server_adaptation"]["use_adversarial"] is False
    assert manifests["A7-noSemantic"]["causal_factors"]["server_semantic_adaptation"] is False
    assert manifests["A7-noSemantic"]["causal_factors"]["server_stage_mmd"] is True
    assert manifests["A7-noStage"]["causal_factors"]["server_semantic_adaptation"] is True
    assert manifests["A7-noStage"]["causal_factors"]["server_stage_mmd"] is False


def test_flower_evaluate_returns_true_cross_entropy() -> None:
    class FixedLogitModel(torch.nn.Module):
        def forward(self, x):
            return x, x, x

    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(
        TensorDataset(logits, labels, torch.zeros(2, 4), torch.zeros(2, dtype=torch.long)),
        batch_size=2,
    )
    cfg = make_config(device="cpu")

    loss, count, metrics = evaluate(FixedLogitModel(), loader, cfg)

    expected = torch.nn.functional.cross_entropy(logits, labels).item()
    assert count == 2
    assert loss == pytest.approx(expected)
    assert metrics["nll"] == pytest.approx(expected)
    assert metrics["accuracy"] == 1.0


def test_replay_teacher_starts_from_second_round(monkeypatch) -> None:
    cfg = make_config(profile="replay_only", seed=42)
    model = create_model(cfg)
    arrays, keys = get_parameters(model)
    client = GapsFlowerClient.__new__(GapsFlowerClient)
    client.client_id = 1
    client.profile = "replay_only"
    client.canonical_profile = "replay_only"
    client.seed = 42
    client.config = cfg
    client.model = model
    client.parameter_keys = keys
    client.gaps_client = type("Stub", (), {"prev_model": object()})()
    client.last_server_state = None
    client.train_samples = 1
    installed: list[dict[str, torch.Tensor]] = []

    monkeypatch.setattr(
        flower_client_module,
        "train_one_round",
        lambda *_args, **_kwargs: (arrays, 1, {}),
    )
    monkeypatch.setattr(
        flower_client_module,
        "set_prev_model_from_state",
        lambda _client, state: installed.append(state),
    )

    client.fit(arrays, {"server_round": 1})
    assert installed == []
    assert client.gaps_client.prev_model is None

    client.fit(arrays, {"server_round": 2})
    assert len(installed) == 1


def test_a1_is_contract_only_and_not_scheduled(tmp_path) -> None:
    manifest = build_run_manifest(
        "A1", 42, repo_root=tmp_path, results_root="results/test"
    )

    assert manifest["contract_only"] is True
    assert manifest["scheduled_for_training"] is False
    assert SPECS["A1"].strategy == "gaps"
    assert SPECS["A1"].use_selective_agg is False
    assert SPECS["A1"].use_domain_adapt is False
    assert manifest["execution_stage"] == "contract_only"


def test_manifest_execution_stages_prevent_one_shot_queueing(tmp_path) -> None:
    core = build_run_manifest("A4S", 42, repo_root=tmp_path, results_root="results/test")
    confirmation = build_run_manifest("A4S", 43, repo_root=tmp_path, results_root="results/test")
    appendix = build_run_manifest("A7-noStage", 42, repo_root=tmp_path, results_root="results/test")

    assert core["execution_stage"] == "core_screening"
    assert confirmation["execution_stage"] == "confirmation"
    assert appendix["execution_stage"] == "appendix_conditional"


def test_generated_remote_shell_scripts_use_lf_only(tmp_path) -> None:
    manifest = build_run_manifest("A0", 42, repo_root=tmp_path, results_root="results/test")
    run_dir = tmp_path / "commands" / manifest["run_name"]

    _write_command_files(run_dir, manifest)

    for name in ("server_command.sh", "client_c1_pi_command.sh"):
        payload = (run_dir / name).read_bytes()
        assert b"\r" not in payload
        assert payload.startswith(b"#!/usr/bin/env bash\nset -euo pipefail\n")


def test_a1_gaps_aggregation_matches_fedavg_when_optional_features_are_off(tmp_path) -> None:
    cfg = make_config(profile="ce_only", seed=42)
    model = create_model(cfg)
    base_arrays, keys = get_parameters(model)
    client_arrays = []
    for offset in (0.01, -0.02):
        client_arrays.append(
            [array + np.asarray(offset, dtype=array.dtype) for array in base_arrays]
        )
    results = []
    for client_id, (arrays, examples) in enumerate(zip(client_arrays, (30, 70)), start=1):
        results.append(
            (
                None,
                FitRes(
                    status=Status(code=Code.OK, message="ok"),
                    parameters=ndarrays_to_parameters(arrays),
                    num_examples=examples,
                    metrics={"client_id": client_id, "num_examples": examples},
                ),
            )
        )

    common = {
        "parameter_keys": keys,
        "reference_state": model.state_dict(),
        "save_history": False,
        "fraction_fit": 1.0,
        "min_fit_clients": 2,
        "min_available_clients": 2,
    }
    fedavg = CheckpointFedAvg(
        output_dir=str(tmp_path / "fedavg"), run_name="A0", **common
    )
    gaps = GapsStrategy(
        output_dir=str(tmp_path / "gaps"),
        run_name="A1",
        use_selective_agg=False,
        use_proto_mmd=False,
        use_domain_adapt=False,
        **common,
    )

    fedavg_parameters, _ = fedavg.aggregate_fit(1, results, [])
    gaps_parameters, _ = gaps.aggregate_fit(1, results, [])

    fedavg_arrays = parameters_to_ndarrays(fedavg_parameters)
    gaps_arrays = parameters_to_ndarrays(gaps_parameters)
    assert len(fedavg_arrays) == len(gaps_arrays)
    for fedavg_array, gaps_array in zip(fedavg_arrays, gaps_arrays):
        np.testing.assert_allclose(fedavg_array, gaps_array, rtol=0.0, atol=1e-7)
