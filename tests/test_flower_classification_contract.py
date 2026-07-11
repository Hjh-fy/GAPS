from collections import OrderedDict

import numpy as np
import pytest
import torch
import flwr as fl
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from gaps_flower.strategy import CheckpointFedAvg, GapsStrategy
from gaps_flower.task import create_model, get_parameters, make_config, set_parameters
from scripts.generate_iotj_classification_ablation_commands import (
    SPECS,
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


@pytest.mark.parametrize("group_id", ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"])
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
    assert manifest["server_adaptation"]["lambda_target_ce"] == 0.0
    assert any("client_5" in arg for arg in manifest["commands"]["server_ecs"])
    assert all("client_3" not in str(command) and "client_4" not in str(command) for command in manifest["commands"].values())
    assert manifest["commands"]["client_c1_pi"][-1] == "42"
    assert "cpu" in manifest["commands"]["client_c1_pi"]
    assert "cpu" in manifest["commands"]["client_c2_pc"]


def test_a1_is_contract_only_and_not_scheduled(tmp_path) -> None:
    manifest = build_run_manifest(
        "A1", 42, repo_root=tmp_path, results_root="results/test"
    )

    assert manifest["contract_only"] is True
    assert manifest["scheduled_for_training"] is False
    assert SPECS["A1"].strategy == "gaps"
    assert SPECS["A1"].use_selective_agg is False
    assert SPECS["A1"].use_domain_adapt is False


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
