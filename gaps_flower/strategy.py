"""Flower strategies for GAPS cloud deployment smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


class CheckpointFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves aggregated server checkpoints each round."""

    def __init__(self, *, parameter_keys: List[str], reference_state: dict, output_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.parameter_keys = parameter_keys
        self.reference_state = reference_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, server_round: int, arrays: NDArrays) -> None:
        state = {}
        for key, value in zip(self.parameter_keys, arrays):
            ref = self.reference_state[key]
            state[key] = torch.tensor(value, dtype=ref.dtype)
        ckpt = {
            "round": int(server_round),
            "model_state": state,
            "parameter_keys": self.parameter_keys,
        }
        path = self.output_dir / f"server_round_{server_round:03d}.pth"
        torch.save(ckpt, path)
        torch.save(ckpt, self.output_dir / "server_latest.pth")
        print(f"[GAPS] Saved aggregated checkpoint: {path}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            arrays = parameters_to_ndarrays(aggregated_parameters)
            self._save_checkpoint(server_round, arrays)
        return aggregated_parameters, aggregated_metrics


def weighted_average(metrics):
    """Weighted average for Flower metric aggregation callbacks."""
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {}
    keys = sorted({key for _, metric in metrics for key in metric.keys()})
    aggregated = {}
    for key in keys:
        weighted_sum = 0.0
        seen = 0
        for num_examples, metric in metrics:
            if key in metric:
                weighted_sum += float(metric[key]) * num_examples
                seen += num_examples
        if seen > 0:
            aggregated[key] = weighted_sum / seen
    return aggregated
