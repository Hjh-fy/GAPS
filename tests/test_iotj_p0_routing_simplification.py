from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from client import Client
from gaps_flower.task import make_config, train_one_round
from scripts.evaluate_iotj_p0_roundwise_routing import (
    LR,
    METHODS,
    ROUNDS,
    STEPS,
    full_hyperparams,
    simple_commission,
)
from scripts.run_iotj_p0_source_fedavg import (
    _checkpoint_index,
    client_argv,
    server_argv,
)


class CountingClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.forward_calls = 0

    def forward(self, x: torch.Tensor):
        self.forward_calls += 1
        features = x.mean(dim=1)
        return self.linear(features), features, features


def test_ce_only_pure_fedavg_contract() -> None:
    server = server_argv("/tmp/p0")
    client = client_argv("python", 1, "/tmp/data")
    assert server[server.index("--strategy") + 1] == "fedavg"
    assert server[server.index("--profile") + 1] == "ce_only"
    assert "--server-calib-data" not in server
    assert client[client.index("--local-epochs") + 1] == "1"
    assert client[client.index("--profile") + 1] == "ce_only"
    assert client[client.index("--proximal-mu") + 1] == "0"


def test_roundwise_checkpoint_count(tmp_path: Path) -> None:
    for round_id in range(1, ROUNDS + 1):
        path = tmp_path / f"server_round_{round_id:03d}.pth"
        path.write_bytes(f"round={round_id}".encode())
    (tmp_path / "server_latest.pth").write_bytes((tmp_path / "server_round_025.pth").read_bytes())
    rows = _checkpoint_index(tmp_path)
    assert len(rows) == 25
    assert rows[-1]["round"] == 25


def test_target_ce_commissioning_disables_all_da_terms() -> None:
    source = inspect.getsource(simple_commission)
    assert "ServerDomainAdaptation" not in source
    assert "F.cross_entropy" in source
    assert STEPS == 100 and LR == 5e-4


def test_full_da_uses_locked_adapter_config() -> None:
    hp = full_hyperparams()
    assert hp["MMD_OBJECTIVE"] == "mmd2"
    assert hp["STAGE_ALIGNMENT"] == "cross_domain_same_class_phase"
    assert hp["ADV_FEATURE_OBJECTIVE"] == "wasserstein_min"
    assert hp["LAMBDA_DEEP_CORAL"] == 0.5
    assert hp["LAMBDA_STAGE_MMD"] == 0.2
    assert hp["LAMBDA_TARGET_CE"] == 0.0
    assert hp["LAMBDA_PROTO_MMD"] == 0.0


def test_each_round_reloads_original_checkpoint() -> None:
    source = inspect.getsource(__import__("scripts.evaluate_iotj_p0_roundwise_routing", fromlist=["main"]).main)
    assert source.count("load_checkpoint_model(str(checkpoint)") == 3
    assert METHODS == ("source_only", "simple_target_ce", "full_target_adapter")


def test_test_split_is_evaluation_only() -> None:
    source = inspect.getsource(__import__("scripts.evaluate_iotj_p0_roundwise_routing", fromlist=["main"]).main)
    assert 'make_loader(data_root, 5, "test"' in source
    assert "test" not in inspect.getsource(simple_commission)


def test_round25_is_fixed_formal_checkpoint() -> None:
    source = inspect.getsource(__import__("scripts.evaluate_iotj_p0_roundwise_routing", fromlist=["main"]).main)
    assert '"formal_comparison_round": 25' in source
    assert 'row["source_round"] != 25' in source


def test_loss_instrumentation_does_not_change_training_contract() -> None:
    config = make_config(device="cpu", local_epochs=1, batch_size=2, profile="ce_only", seed=42)
    config.NUM_CLASSES = 2
    model = CountingClassifier()
    x = torch.arange(24, dtype=torch.float32).view(4, 2, 3) / 24.0
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    reg = torch.zeros((4, 4), dtype=torch.float32)
    phase = torch.zeros(4, dtype=torch.long)
    client = Client(client_id=1, config=config); client.set_model(model)
    client.update_dataloader(DataLoader(TensorDataset(x, y, reg, phase), batch_size=2))
    _arrays, examples, metrics = train_one_round(client, 1)
    assert model.forward_calls == 2
    assert examples == 4
    assert metrics["train_metric_examples"] == 4
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert metrics["train_ce_averaging"] == "sample_weighted_over_local_minibatches"
