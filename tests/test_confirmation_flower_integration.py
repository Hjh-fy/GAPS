from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import gaps_flower.client_app as client_app
import gaps_flower.server_app as server_app
import gaps_flower.strategy as strategy_module
from gaps_flower.client_app import GapsFlowerClient
from gaps_flower.flower_message_audit import audit_fit_ins
from gaps_flower.observability import JsonlObserver, ObserverIdentity
from gaps_flower.strategy import CheckpointFedAvg, GapsStrategy
from gaps_flower.task import get_parameters


VOLATILE_FLOWER_FIELDS = {
    ("metrics", "fit_seconds"),
    ("metrics", "evaluate_seconds"),
}


class FakeClientProxy(ClientProxy):
    def get_properties(self, *args, **kwargs):
        raise NotImplementedError

    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def reconnect(self, *args, **kwargs):
        raise NotImplementedError


def make_identity(*, producer: str, client_id: str | None = None) -> ObserverIdentity:
    return ObserverIdentity(
        run_id="c12_to_c5__b2__s42",
        attempt_id="c12_to_c5__b2__s42__a001",
        group_id="B2",
        training_seed=42,
        client_id=client_id,
        host_id="ecs" if producer == "server" else "edge-c1",
        producer=producer,
        confirmation_commit="a" * 40,
        source_archive_sha256="b" * 64,
        dataset_manifest_sha256="c" * 64,
        algorithm_config_sha256="d" * 64,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def event_rows(path: Path, event_type: str) -> list[dict[str, object]]:
    return [row for row in read_jsonl(path) if row["event_type"] == event_type]


def test_client_observer_preserves_flower_fields_and_emits_fit_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        LOCAL_EPOCHS=1,
        USE_REPLAY_DISTILL=False,
        USE_ALIGN=False,
        USE_PROTO_DECOUPLING=False,
    )
    loader = SimpleNamespace(dataset=[0, 1, 2])
    monkeypatch.setattr(client_app, "make_config", lambda **_kwargs: config)
    monkeypatch.setattr(client_app, "create_model", lambda _config: torch.nn.Linear(2, 1))
    monkeypatch.setattr(
        client_app, "load_client_loaders", lambda *_args, **_kwargs: (loader, loader)
    )
    monkeypatch.setattr(
        client_app, "make_client", lambda *_args, **_kwargs: SimpleNamespace()
    )

    returned_arrays = [
        np.asarray([[1.25, -2.5]], dtype=np.float32),
        np.asarray([0.75], dtype=np.float32),
    ]
    base_metrics = {"loss": 0.5, "prototype_json": '{"0,0":[1.0,2.0]}'}
    train_calls: list[tuple[object, int, dict[str, object]]] = []

    def fake_train(gaps_client, round_idx, *, fit_config):
        train_calls.append((gaps_client, round_idx, fit_config))
        return [array.copy() for array in returned_arrays], 3, dict(base_metrics)

    monkeypatch.setattr(client_app, "train_one_round", fake_train)
    on_events = tmp_path / "client-on.jsonl"
    on_observer = JsonlObserver(
        make_identity(producer="client", client_id="C1"), on_events
    )
    off_client = GapsFlowerClient(
        client_id=1,
        data_root="unused",
        device="cpu",
        local_epochs=1,
        batch_size=3,
    )
    on_client = GapsFlowerClient(
        client_id=1,
        data_root="unused",
        device="cpu",
        local_epochs=1,
        batch_size=3,
        observer=on_observer,
    )
    input_arrays, _keys = get_parameters(off_client.model)
    off_config = {"server_round": 1}
    on_config = {"server_round": 1}

    off_arrays, off_n, off_metrics = off_client.fit(input_arrays, off_config)
    on_arrays, on_n, on_metrics = on_client.fit(input_arrays, on_config)
    on_observer.close()

    assert VOLATILE_FLOWER_FIELDS == {
        ("metrics", "fit_seconds"),
        ("metrics", "evaluate_seconds"),
    }
    assert len(off_arrays) == len(on_arrays)
    for left, right in zip(off_arrays, on_arrays):
        np.testing.assert_array_equal(left, right)
    assert off_n == on_n
    assert set(off_metrics) == set(on_metrics)
    for key in off_metrics:
        if ("metrics", key) not in VOLATILE_FLOWER_FIELDS:
            assert off_metrics[key] == on_metrics[key]
    assert not any(key.startswith("observer") for key in on_metrics)
    assert off_config == on_config == {"server_round": 1}
    assert len(train_calls) == 2
    assert all(round_idx == 1 for _client, round_idx, _fit_config in train_calls)
    assert all(
        fit_config in (off_config, on_config)
        for _client, _round_idx, fit_config in train_calls
    )

    rows = read_jsonl(on_events)
    assert {row["event_type"] for row in rows} >= {
        "client_fit_start",
        "client_train_start",
        "client_train_end",
        "client_fit_end",
    }
    lifecycle = [
        row["event_type"]
        for row in rows
        if row["event_type"] != "observer_overhead"
    ]
    assert lifecycle == [
        "client_fit_start",
        "client_train_start",
        "client_train_end",
        "client_fit_end",
    ]
    train_end = event_rows(on_events, "client_train_end")[0]
    assert train_end["payload"]["client_train_core_ns"] >= 0
    fit_end = event_rows(on_events, "client_fit_end")[0]
    assert fit_end["payload"]["client_fit_callback_ns"] >= 0


def test_legacy_client_built_with_new_uses_null_observer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(2, 1)
    arrays, keys = get_parameters(model)
    client = GapsFlowerClient.__new__(GapsFlowerClient)
    client.client_id = 1
    client.profile = "smoke"
    client.canonical_profile = "ce_only"
    client.seed = 42
    client.config = SimpleNamespace(
        LOCAL_EPOCHS=1,
        USE_REPLAY_DISTILL=False,
        USE_ALIGN=False,
        USE_PROTO_DECOUPLING=False,
    )
    client.model = model
    client.parameter_keys = keys
    client.gaps_client = SimpleNamespace()
    client.last_server_state = None
    client.train_samples = 1
    monkeypatch.setattr(
        client_app,
        "train_one_round",
        lambda *_args, **_kwargs: ([array.copy() for array in arrays], 1, {}),
    )

    returned_arrays, num_examples, metrics = client.fit(
        arrays, {"server_round": 1}
    )

    for left, right in zip(returned_arrays, arrays):
        np.testing.assert_array_equal(left, right)
    assert num_examples == 1
    assert not any(key.startswith("observer") for key in metrics)


def _strategy(
    output_dir: Path,
    arrays: list[np.ndarray],
    keys: list[str],
    reference_state: dict[str, torch.Tensor],
    *,
    observer=None,
) -> GapsStrategy:
    strategy = GapsStrategy(
        parameter_keys=keys,
        reference_state=reference_state,
        output_dir=str(output_dir),
        run_name=output_dir.name,
        save_history=False,
        use_selective_agg=False,
        use_proto_mmd=False,
        use_domain_adapt=False,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
        initial_parameters=ndarrays_to_parameters(arrays),
        observer=observer,
    )
    strategy.semantic_protos = {"0,0": torch.tensor([1.0, 2.0])}
    return strategy


def _fit_results(arrays: list[np.ndarray]) -> list[tuple[FakeClientProxy, FitRes]]:
    results = []
    for client_id, (offset, examples) in enumerate(
        zip((0.25, -0.5), (3, 7)), start=1
    ):
        client_arrays = [
            array + np.asarray(offset, dtype=array.dtype) for array in arrays
        ]
        results.append(
            (
                FakeClientProxy(f"proxy-{client_id}"),
                FitRes(
                    status=Status(code=Code.OK, message="ok"),
                    parameters=ndarrays_to_parameters(client_arrays),
                    num_examples=examples,
                    metrics={
                        "client_id": client_id,
                        "num_examples": examples,
                        "score": float(client_id),
                    },
                ),
            )
        )
    return results


def _install_flower_configure_fit(
    monkeypatch: pytest.MonkeyPatch, proxies: list[FakeClientProxy]
) -> None:
    def fake_configure_fit(_self, _server_round, parameters, _client_manager):
        return [
            (proxy, FitIns(parameters=parameters, config={"existing": "stable"}))
            for proxy in proxies
        ]

    monkeypatch.setattr(
        strategy_module.fl.server.strategy.FedAvg,
        "configure_fit",
        fake_configure_fit,
    )


def test_server_observer_audits_final_messages_and_preserves_aggregate_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 1)
    arrays, keys = get_parameters(model)
    results = _fit_results(arrays)
    proxies = [proxy for proxy, _fit_res in results]
    _install_flower_configure_fit(monkeypatch, proxies)

    on_events = tmp_path / "server-on.jsonl"
    on_observer = JsonlObserver(make_identity(producer="server"), on_events)
    off_strategy = _strategy(
        tmp_path / "off", arrays, keys, model.state_dict()
    )
    on_strategy = _strategy(
        tmp_path / "on", arrays, keys, model.state_dict(), observer=on_observer
    )
    for strategy in (off_strategy, on_strategy):
        strategy.use_domain_adapt = True
        strategy.domain_adapt_warmup = 0
        strategy._run_domain_adapt = (
            lambda _round, _state, plain_arrays, _results, _weights: (
                "adapted.pth",
                {"executed": True},
                plain_arrays,
            )
        )

    initial_parameters = ndarrays_to_parameters(arrays)
    off_configured = off_strategy.configure_fit(1, initial_parameters, object())
    on_configured = on_strategy.configure_fit(1, initial_parameters, object())
    off_parameters, off_metrics = off_strategy.aggregate_fit(1, results, [])
    on_parameters, on_metrics = on_strategy.aggregate_fit(1, results, [])
    on_observer.close()

    assert CheckpointFedAvg.configure_fit is strategy_module.fl.server.strategy.FedAvg.configure_fit
    assert [fit_ins.config for _proxy, fit_ins in off_configured] == [
        fit_ins.config for _proxy, fit_ins in on_configured
    ]
    for _proxy, fit_ins in on_configured:
        assert fit_ins.config["existing"] == "stable"
        assert fit_ins.config["server_round"] == 1
        assert fit_ins.config["semantic_proto_ready"] is True
        assert fit_ins.config["semantic_proto_count"] == 1
        assert "semantic_protos_json" in fit_ins.config

    off_arrays = parameters_to_ndarrays(off_parameters)
    on_arrays = parameters_to_ndarrays(on_parameters)
    assert len(off_arrays) == len(on_arrays)
    for left, right in zip(off_arrays, on_arrays):
        np.testing.assert_array_equal(left, right)
    assert off_metrics == on_metrics

    fitins_rows = event_rows(on_events, "flower_fitins_prepared")
    assert {row["payload"]["proxy_id"] for row in fitins_rows} == {
        "proxy-1",
        "proxy-2",
    }
    configured_by_proxy = {proxy.cid: fit_ins for proxy, fit_ins in on_configured}
    for row in fitins_rows:
        expected_audit = audit_fit_ins(configured_by_proxy[row["payload"]["proxy_id"]])
        downlink = row["payload"]["downlink_audit"]
        assert downlink["application_message_bytes"] == expected_audit.application_message_bytes
        assert (
            downlink["application_message_sha256"]
            == expected_audit.application_message_sha256
        )
        assert downlink["logical"] == expected_audit.logical

    fitres_rows = event_rows(on_events, "flower_fitres_available")
    assert {row["client_id"] for row in fitres_rows} == {"C1", "C2"}
    assert {row["payload"]["proxy_id"] for row in fitres_rows} == {
        "proxy-1",
        "proxy-2",
    }
    assert all("uplink_audit" in row["payload"] for row in fitres_rows)
    assert len(event_rows(on_events, "server_da_start")) == 1
    assert len(event_rows(on_events, "server_da_end")) == 1
    aggregate_end = event_rows(on_events, "server_aggregate_end")[0]["payload"]
    fit_round_end = event_rows(on_events, "fit_round_end")[0]["payload"]
    assert aggregate_end["da_executed"] is True
    assert (
        aggregate_end["server_aggregate_fit_total_ns"]
        >= aggregate_end["server_da_total_ns"]
        >= 0
    )
    assert aggregate_end["server_aggregate_non_da_ns"] == (
        aggregate_end["server_aggregate_fit_total_ns"]
        - aggregate_end["server_da_total_ns"]
    )
    assert (
        fit_round_end["fit_round_wall_ns"]
        >= aggregate_end["server_aggregate_fit_total_ns"]
    )


def test_da_events_are_absent_when_domain_adaptation_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 1)
    arrays, keys = get_parameters(model)
    results = _fit_results(arrays)
    proxies = [proxy for proxy, _fit_res in results]
    _install_flower_configure_fit(monkeypatch, proxies)
    events = tmp_path / "server-no-da.jsonl"
    observer = JsonlObserver(make_identity(producer="server"), events)
    strategy = _strategy(
        tmp_path / "no-da", arrays, keys, model.state_dict(), observer=observer
    )

    strategy.configure_fit(1, ndarrays_to_parameters(arrays), object())
    strategy.aggregate_fit(1, results, [])
    observer.close()

    event_types = {row["event_type"] for row in read_jsonl(events)}
    assert "server_da_start" not in event_types
    assert "server_da_end" not in event_types
    aggregate_end = event_rows(events, "server_aggregate_end")[0]["payload"]
    assert aggregate_end["da_executed"] is False
    assert aggregate_end["server_da_total_ns"] == 0


def test_null_observer_skips_message_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 1)
    arrays, keys = get_parameters(model)
    results = _fit_results(arrays)
    proxies = [proxy for proxy, _fit_res in results]
    _install_flower_configure_fit(monkeypatch, proxies)
    strategy = _strategy(tmp_path / "null", arrays, keys, model.state_dict())

    def forbidden_audit(*_args, **_kwargs):
        raise AssertionError("NullObserver must not trigger Flower serialization")

    monkeypatch.setattr(strategy_module, "audit_fit_ins", forbidden_audit)
    monkeypatch.setattr(strategy_module, "audit_fit_res", forbidden_audit)

    strategy.configure_fit(1, ndarrays_to_parameters(arrays), object())
    strategy.aggregate_fit(1, results, [])


@pytest.mark.parametrize("module_name", ["gaps_flower.client_app", "gaps_flower.server_app"])
def test_flower_cli_exposes_observer_paths(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--observer-context" in result.stdout
    assert "--observer-events" in result.stdout


@pytest.mark.parametrize("app", ["client", "server"])
def test_flower_apps_close_observer_when_flower_runtime_raises(
    app: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingObserver:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    observer = RecordingObserver()
    load_calls: list[tuple[str | None, str | None]] = []

    def fake_load(context_path, events_path):
        load_calls.append((context_path, events_path))
        return observer

    context = str(tmp_path / "context.json")
    events = str(tmp_path / "events.jsonl")
    if app == "client":
        monkeypatch.setattr(client_app, "load_observer", fake_load, raising=False)
        constructed: list[dict[str, object]] = []
        monkeypatch.setattr(
            client_app,
            "GapsFlowerClient",
            lambda **kwargs: constructed.append(kwargs) or object(),
        )
        monkeypatch.setattr(
            client_app.fl.client,
            "start_numpy_client",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("client stopped")),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "client_app",
                "--client-id",
                "1",
                "--data-root",
                "unused",
                "--observer-context",
                context,
                "--observer-events",
                events,
            ],
        )
        with pytest.raises(RuntimeError, match="client stopped"):
            client_app.main()
        assert constructed[0]["observer"] is observer
    else:
        monkeypatch.setattr(server_app, "load_observer", fake_load, raising=False)
        monkeypatch.setattr(server_app, "save_run_config", lambda *_args: None)
        monkeypatch.setattr(
            server_app, "validate_domain_adaptation_request", lambda *_args: None
        )
        monkeypatch.setattr(server_app, "make_config", lambda **_kwargs: object())
        model = torch.nn.Linear(1, 1)
        monkeypatch.setattr(server_app, "create_model", lambda _config: model)
        monkeypatch.setattr(
            server_app,
            "get_parameters",
            lambda _model: ([np.asarray([1.0], dtype=np.float32)], ["weight"]),
        )
        constructed = []
        monkeypatch.setattr(
            server_app,
            "CheckpointFedAvg",
            lambda **kwargs: constructed.append(kwargs) or object(),
        )
        monkeypatch.setattr(
            server_app.fl.server,
            "start_server",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("server stopped")),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "server_app",
                "--output-dir",
                str(tmp_path / "output"),
                "--observer-context",
                context,
                "--observer-events",
                events,
            ],
        )
        with pytest.raises(RuntimeError, match="server stopped"):
            server_app.main()
        assert constructed[0]["observer"] is observer

    assert load_calls == [(context, events)]
    assert observer.close_count == 1
