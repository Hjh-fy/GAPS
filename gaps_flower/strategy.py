"""Flower strategies for GAPS cloud deployment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


class CheckpointFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves checkpoints, history, and client statistics."""

    def __init__(
        self,
        *,
        parameter_keys: List[str],
        reference_state: dict,
        output_dir: str,
        run_name: str = "",
        save_history: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parameter_keys = parameter_keys
        self.reference_state = reference_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.save_history = save_history
        self._round_events = {}
        self.history_path = self.output_dir / "history.json"

    def _round_event(self, server_round: int) -> dict:
        return self._round_events.setdefault(
            int(server_round),
            {"round": int(server_round), "run_name": self.run_name},
        )

    def _write_history(self) -> None:
        if not self.save_history:
            return
        ordered = [self._round_events[key] for key in sorted(self._round_events)]
        payload = {"run_name": self.run_name, "rounds": ordered}
        self.history_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

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

    @staticmethod
    def _parse_json_metric(value, default):
        if not isinstance(value, str) or not value:
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _collect_client_fit_stats(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]
    ) -> tuple[str, dict]:
        clients = []
        weighted_feature = None
        weighted_examples = 0.0

        for _proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            client_id = int(metrics.get("client_id", -1))
            num_examples = int(metrics.get("num_examples", fit_res.num_examples))
            counts = self._parse_json_metric(metrics.get("class_phase_counts_json"), {})
            global_feature = self._parse_json_metric(metrics.get("global_feature_json"), [])

            client_entry = {
                "client_id": client_id,
                "num_examples": num_examples,
                "proto_examples": int(metrics.get("proto_examples", 0)),
                "local_epochs": int(metrics.get("local_epochs", 0)),
                "fit_seconds": float(metrics.get("fit_seconds", 0.0)),
                "feature_norm": float(metrics.get("feature_norm", 0.0)),
                "feature_mean": float(metrics.get("feature_mean", 0.0)),
                "feature_std": float(metrics.get("feature_std", 0.0)),
                "class_phase_counts": counts,
            }
            if global_feature:
                client_entry["global_feature_dim"] = len(global_feature)
                feature_tensor = torch.tensor(global_feature, dtype=torch.float64)
                if weighted_feature is None:
                    weighted_feature = torch.zeros_like(feature_tensor)
                weighted_feature += feature_tensor * max(num_examples, 0)
                weighted_examples += max(float(num_examples), 0.0)
            clients.append(client_entry)

        global_summary = {}
        if weighted_feature is not None and weighted_examples > 0:
            global_feature_mean = weighted_feature / weighted_examples
            global_summary = {
                "global_feature_dim": int(global_feature_mean.numel()),
                "global_feature_norm": float(torch.norm(global_feature_mean, p=2).item()),
                "global_feature_mean": float(global_feature_mean.mean().item()),
                "global_feature_std": float(global_feature_mean.std(unbiased=False).item()),
                "global_feature_vector": global_feature_mean.tolist(),
            }

        payload = {
            "run_name": self.run_name,
            "round": int(server_round),
            "clients": sorted(clients, key=lambda item: item["client_id"]),
            "global_summary": global_summary,
        }
        path = self.output_dir / f"client_stats_round_{server_round:03d}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[GAPS] Saved client stats: {path}")
        return str(path), global_summary

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        event = self._round_event(server_round)
        event["fit_clients"] = len(results)
        event["fit_failures"] = len(failures)
        event["fit_metrics"] = dict(aggregated_metrics or {})
        if results:
            stats_path, global_summary = self._collect_client_fit_stats(server_round, results)
            event["client_stats"] = stats_path
            event["global_feature_summary"] = {
                key: value for key, value in global_summary.items() if key != "global_feature_vector"
            }
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
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        event = self._round_event(server_round)
        event["evaluate_clients"] = len(results)
        event["evaluate_failures"] = len(failures)
        event["evaluate_loss"] = float(aggregated_loss) if aggregated_loss is not None else None
        event["evaluate_metrics"] = dict(aggregated_metrics or {})
        self._write_history()
        return aggregated_loss, aggregated_metrics


class GapsStrategy(CheckpointFedAvg):
    """Placeholder GAPS strategy.

    It currently keeps the checkpoint/statistics FedAvg behavior while reserving
    a stable CLI switch for later custom aggregation and server-side adaptation.
    """


def weighted_average(metrics):
    """Weighted average for numeric Flower metrics."""
    if not metrics:
        return {}
    skip_keys = {
        "client_id",
        "round",
        "class_phase_counts_json",
        "global_feature_json",
    }
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
