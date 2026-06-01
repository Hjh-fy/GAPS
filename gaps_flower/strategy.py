"""Flower strategies for GAPS cloud deployment smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


class CheckpointFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves aggregated checkpoints and history."""

    def __init__(self, *, parameter_keys: List[str], reference_state: dict, output_dir: str, run_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.parameter_keys = parameter_keys
        self.reference_state = reference_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.history = []
        self._round_events = {}
        self.history_path = self.output_dir / "history.json"

    def _round_event(self, server_round: int) -> dict:
        event = self._round_events.setdefault(
            int(server_round),
            {"round": int(server_round), "run_name": self.run_name},
        )
        return event

    def _write_history(self) -> None:
        ordered = [self._round_events[key] for key in sorted(self._round_events)]
        payload = {
            "run_name": self.run_name,
            "rounds": ordered,
        }
        self.history_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _save_checkpoint(self, server_round: int, arrays: NDArrays) -> str:
        state = {}
        for key, value in zip(self.parameter_keys, arrays):
            ref = self.reference_state[key]
            state[key] = torch.tensor(value, dtype=ref.dtype)
        ckpt = {
            "round": int(server_round),
            "model_state": state,
            "parameter_keys": self.parameter_keys,
            "run_name": self.run_name,
        }
        path = self.output_dir / f"server_round_{server_round:03d}.pth"
        torch.save(ckpt, path)
        torch.save(ckpt, self.output_dir / "server_latest.pth")
        print(f"[GAPS] Saved aggregated checkpoint: {path}")
        return str(path)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        event = self._round_event(server_round)
        event["fit_clients"] = len(results)
        event["fit_failures"] = len(failures)
        event["fit_metrics"] = dict(aggregated_metrics or {})
        if aggregated_parameters is not None:
            arrays = parameters_to_ndarrays(aggregated_parameters)
            event["checkpoint"] = self._save_checkpoint(server_round, arrays)
        self._write_history()
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], dict]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        event = self._round_event(server_round)
        event["evaluate_clients"] = len(results)
        event["evaluate_failures"] = len(failures)
        event["evaluate_loss"] = float(aggregated_loss) if aggregated_loss is not None else None
        event["evaluate_metrics"] = dict(aggregated_metrics or {})
        self._write_history()
        return aggregated_loss, aggregated_metrics


def weighted_average(metrics):
    """Weighted average for numeric Flower metrics."""
    if not metrics:
        return {}
    skip_keys = {"client_id", "round"}
    keys = sorted({key for _, metric in metrics for key in metric.keys() if key not in skip_keys})
    aggregated = {}
    for key in keys:
        weighted_sum = 0.0
        seen = 0
        for num_examples, metric in metrics:
            if key not in metric:
                continue
            value = metric[key]
            if not isinstance(value, (int, float)):
                continue
            weighted_sum += float(value) * num_examples
            seen += num_examples
        if seen > 0:
            aggregated[key] = weighted_sum / seen
    return aggregated
