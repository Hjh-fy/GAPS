"""Flower strategies for GAPS cloud deployment runs."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import EvaluateRes, FitRes, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
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

    @staticmethod
    def _prototype_summary(prototypes: dict) -> dict:
        if not prototypes:
            return {"num_global_prototypes": 0}
        tensors = [torch.tensor(value, dtype=torch.float64) for value in prototypes.values()]
        norms = [float(torch.norm(tensor, p=2).item()) for tensor in tensors]
        dims = sorted({int(tensor.numel()) for tensor in tensors})
        return {
            "num_global_prototypes": len(prototypes),
            "prototype_dims": dims,
            "prototype_norm_mean": float(sum(norms) / len(norms)),
            "prototype_norm_min": float(min(norms)),
            "prototype_norm_max": float(max(norms)),
        }

    @staticmethod
    def _prototype_var_summary(proto_vars: dict) -> dict:
        if not proto_vars:
            return {"num_global_proto_vars": 0}
        tensors = [torch.tensor(value, dtype=torch.float64) for value in proto_vars.values()]
        means = [float(tensor.mean().item()) for tensor in tensors]
        dims = sorted({int(tensor.numel()) for tensor in tensors})
        return {
            "num_global_proto_vars": len(proto_vars),
            "prototype_var_dims": dims,
            "prototype_var_mean_mean": float(sum(means) / len(means)),
            "prototype_var_mean_min": float(min(means)),
            "prototype_var_mean_max": float(max(means)),
        }

    def _collect_client_fit_stats(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]
    ) -> tuple[str, dict, str, dict]:
        clients = []
        weighted_feature = None
        weighted_examples = 0.0
        proto_sums = {}
        proto_counts = {}
        var_sums = {}
        var_counts = {}
        client_proto_keys = {}

        for _proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            client_id = int(metrics.get("client_id", -1))
            num_examples = int(metrics.get("num_examples", fit_res.num_examples))
            counts = self._parse_json_metric(metrics.get("class_phase_counts_json"), {})
            global_feature = self._parse_json_metric(metrics.get("global_feature_json"), [])
            prototypes = self._parse_json_metric(metrics.get("prototype_json"), {})
            proto_vars = self._parse_json_metric(metrics.get("prototype_var_json"), {})

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
                "prototype_count": int(metrics.get("prototype_count", len(prototypes))),
                "prototype_var_count": int(metrics.get("prototype_var_count", len(proto_vars))),
            }
            if global_feature:
                client_entry["global_feature_dim"] = len(global_feature)
                feature_tensor = torch.tensor(global_feature, dtype=torch.float64)
                if weighted_feature is None:
                    weighted_feature = torch.zeros_like(feature_tensor)
                weighted_feature += feature_tensor * max(num_examples, 0)
                weighted_examples += max(float(num_examples), 0.0)

            client_proto_keys[str(client_id)] = sorted(prototypes.keys())
            for key, vector in prototypes.items():
                count = int(counts.get(key, 0))
                if count <= 0:
                    continue
                tensor = torch.tensor(vector, dtype=torch.float64)
                if key not in proto_sums:
                    proto_sums[key] = torch.zeros_like(tensor)
                    proto_counts[key] = 0
                proto_sums[key] += tensor * count
                proto_counts[key] += count
            for key, vector in proto_vars.items():
                count = int(counts.get(key, 0))
                if count <= 0:
                    continue
                tensor = torch.tensor(vector, dtype=torch.float64)
                if key not in var_sums:
                    var_sums[key] = torch.zeros_like(tensor)
                    var_counts[key] = 0
                var_sums[key] += tensor * count
                var_counts[key] += count
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

        global_prototypes = {}
        prototype_counts = {}
        for key, proto_sum in sorted(proto_sums.items()):
            count = proto_counts.get(key, 0)
            if count <= 0:
                continue
            global_prototypes[key] = (proto_sum / count).tolist()
            prototype_counts[key] = int(count)

        global_proto_vars = {}
        prototype_var_counts = {}
        for key, var_sum in sorted(var_sums.items()):
            count = var_counts.get(key, 0)
            if count <= 0:
                continue
            global_proto_vars[key] = (var_sum / count).tolist()
            prototype_var_counts[key] = int(count)

        client_stats_payload = {
            "run_name": self.run_name,
            "round": int(server_round),
            "clients": sorted(clients, key=lambda item: item["client_id"]),
            "global_summary": global_summary,
        }
        client_stats_path = self.output_dir / f"client_stats_round_{server_round:03d}.json"
        client_stats_path.write_text(
            json.dumps(client_stats_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[GAPS] Saved client stats: {client_stats_path}")

        prototype_summary = self._prototype_summary(global_prototypes)
        var_summary = self._prototype_var_summary(global_proto_vars)
        prototype_summary.update(var_summary)
        prototype_payload = {
            "run_name": self.run_name,
            "round": int(server_round),
            "global_prototypes": global_prototypes,
            "prototype_counts": prototype_counts,
            "global_proto_vars": global_proto_vars,
            "prototype_var_counts": prototype_var_counts,
            "client_proto_keys": client_proto_keys,
            "summary": prototype_summary,
        }
        prototype_stats_path = self.output_dir / f"prototype_stats_round_{server_round:03d}.json"
        prototype_stats_path.write_text(
            json.dumps(prototype_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[GAPS] Saved prototype stats: {prototype_stats_path}")

        return str(client_stats_path), global_summary, str(prototype_stats_path), prototype_summary

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
        event["aggregation_method"] = "fedavg"
        if results:
            stats_path, global_summary, proto_path, proto_summary = self._collect_client_fit_stats(server_round, results)
            event["client_stats"] = stats_path
            event["prototype_stats"] = proto_path
            event["global_feature_summary"] = {
                key: value for key, value in global_summary.items() if key != "global_feature_vector"
            }
            event["prototype_summary"] = proto_summary
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
    """GAPS 自定义聚合策略

    将 Flower 客户端 Parameters 转为 state_dict，
    使用 GAPS 的 _aggregate_params_gaps 进行加权聚合，再转回 Flower Parameters。

    阶段 4.1：语义原型 EMA 聚合
    - 每轮从 client 上传的 class-phase μ_cp / σ²_cp 做加权平均后，
      通过 EMA 更新全局语义原型 semantic_protos
    - 不依赖 Server 类（避免引入 val_loader/test_loader）
    - 云端仅接收统计量，符合联邦学习隐私叙事

    Attributes:
        proto_ema_alpha: 原型 EMA 平滑系数（默认 0.8）
        semantic_protos: EMA 全局语义原型 μ^{sem}_cp
        semantic_proto_vars: EMA 全局语义方差 σ²_cp
    """

    def __init__(
        self,
        *,
        proto_ema_alpha: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proto_ema_alpha = proto_ema_alpha
        self.semantic_protos: Dict[str, torch.Tensor] = {}
        self.semantic_proto_vars: Dict[str, torch.Tensor] = {}

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], dict]:
        if not results:
            return None, {}

        # ── 1. 每个客户端: Flower Parameters → state_dict(OrderedDict) ──
        client_state_dicts: List[OrderedDict] = []
        num_examples_list: List[int] = []
        for _proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            state = OrderedDict()
            for key, arr in zip(self.parameter_keys, arrays):
                ref = self.reference_state.get(key)
                if ref is not None:
                    state[key] = torch.tensor(arr, dtype=ref.dtype)
                else:
                    state[key] = torch.tensor(arr)
            client_state_dicts.append(state)
            num_examples_list.append(fit_res.num_examples)

        # ── 2. 计算聚合权重 w_i = n_i / Σn_j ──
        total_examples = sum(num_examples_list)
        weights = torch.tensor(
            [n / total_examples for n in num_examples_list],
            dtype=torch.float32,
        )

        # ── 3. GAPS 加权参数聚合 ──
        aggregated_state = self._aggregate_params_gaps(
            client_state_dicts, weights
        )

        # ── 4. state_dict → Flower Parameters ──
        aggregated_ndarrays: List[np.ndarray] = []
        for key in self.parameter_keys:
            agg_tensor = aggregated_state[key]
            aggregated_ndarrays.append(agg_tensor.detach().cpu().numpy())
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)

        # ── 5. 聚合 metrics（复用加权平均） ──
        metrics_pairs = [
            (fit_res.num_examples, fit_res.metrics)
            for _, fit_res in results
        ]
        aggregated_metrics = weighted_average(metrics_pairs)

        # ── 6. 记录 history + 保存统计 ──
        event = self._round_event(server_round)
        event["fit_clients"] = len(results)
        event["fit_failures"] = len(failures)
        event["fit_metrics"] = dict(aggregated_metrics or {})
        event["aggregation_method"] = "gaps"
        if results:
            stats_path, global_summary, proto_path, proto_summary = (
                self._collect_client_fit_stats(server_round, results)
            )
            event["client_stats"] = stats_path
            event["prototype_stats"] = proto_path
            event["global_feature_summary"] = {
                key: value
                for key, value in global_summary.items()
                if key != "global_feature_vector"
            }
            event["prototype_summary"] = proto_summary

            # ── 阶段 4.1: EMA 语义原型更新 ──
            self._update_semantic_protos(server_round)
            semantic_path = self._save_semantic_protos(server_round)
            event["semantic_protos"] = semantic_path

        if aggregated_parameters is not None:
            arrays = parameters_to_ndarrays(aggregated_parameters)
            event["checkpoint"] = self._save_checkpoint(server_round, arrays)
        self._write_history()
        return aggregated_parameters, aggregated_metrics

    @staticmethod
    def _aggregate_params_gaps(
        client_params: List[OrderedDict],
        weights: torch.Tensor,
    ) -> OrderedDict:
        """GAPS 加权参数聚合

        与 FedAvg 数学上等价（按样本数加权平均），
        但走 GAPS 代码路径，后续可扩展为:
          - 选择性聚合 (selective_weights)
          - 可学习聚合 (learnable_aggregate)
          - 原型解耦聚合 (prototype_decoupling)

        Args:
            client_params: 各客户端的 state_dict 列表
            weights: 客户端聚合权重 (n_i / Σn_j)

        Returns:
            聚合后的 OrderedDict
        """
        reference_keys = list(client_params[0].keys())
        agg_params = OrderedDict()
        for key in reference_keys:
            first_param = client_params[0][key]
            agg_params[key] = torch.zeros_like(
                first_param, dtype=torch.float32
            )
            for i, params in enumerate(client_params):
                if key in params:
                    agg_params[key] += weights[i] * params[key].float()
        return agg_params

    def _update_semantic_protos(self, server_round: int) -> None:
        """从已保存的 prototype_stats JSON 读取当轮加权原型，做 EMA 更新

        数学公式:
          μ^{sem}_t = α · μ^{sem}_{t-1}  +  (1 − α) · μ^{weighted}_t
          σ²_t      = α · σ²_{t-1}       +  (1 − α) · σ²^{weighted}_t

        其中 α = proto_ema_alpha，首轮直接初始化为 μ^{weighted}_1

        Args:
            server_round: 当前联邦轮次
        """
        proto_path = self.output_dir / f"prototype_stats_round_{server_round:03d}.json"
        if not proto_path.exists():
            return

        data = json.loads(proto_path.read_text(encoding="utf-8"))
        round_protos = data.get("global_prototypes", {})
        round_vars = data.get("global_proto_vars", {})
        alpha = self.proto_ema_alpha

        if not self.semantic_protos:
            # 首轮直接初始化
            for key, vec in round_protos.items():
                self.semantic_protos[key] = torch.tensor(vec, dtype=torch.float32)
            for key, vec in round_vars.items():
                self.semantic_proto_vars[key] = torch.tensor(vec, dtype=torch.float32)
        else:
            for key, vec in round_protos.items():
                new_vec = torch.tensor(vec, dtype=torch.float32)
                if key in self.semantic_protos:
                    self.semantic_protos[key] = (
                        alpha * self.semantic_protos[key] + (1.0 - alpha) * new_vec
                    )
                else:
                    self.semantic_protos[key] = new_vec
            for key, vec in round_vars.items():
                new_vec = torch.tensor(vec, dtype=torch.float32)
                if key in self.semantic_proto_vars:
                    self.semantic_proto_vars[key] = (
                        alpha * self.semantic_proto_vars[key] + (1.0 - alpha) * new_vec
                    )
                else:
                    self.semantic_proto_vars[key] = new_vec

    def _save_semantic_protos(self, server_round: int) -> str:
        """保存 EMA 语义原型到独立 JSON 文件

        Args:
            server_round: 当前联邦轮次

        Returns:
            保存的文件路径
        """
        proto_serializable = {}
        for key, tensor in self.semantic_protos.items():
            proto_serializable[key] = tensor.detach().cpu().tolist()
        var_serializable = {}
        for key, tensor in self.semantic_proto_vars.items():
            var_serializable[key] = tensor.detach().cpu().tolist()

        payload = {
            "run_name": self.run_name,
            "round": int(server_round),
            "proto_ema_alpha": self.proto_ema_alpha,
            "num_semantic_protos": len(proto_serializable),
            "num_semantic_proto_vars": len(var_serializable),
            "semantic_protos": proto_serializable,
            "semantic_proto_vars": var_serializable,
        }
        path = self.output_dir / f"semantic_protos_round_{server_round:03d}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest = self.output_dir / "semantic_protos_latest.json"
        latest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[GAPS] Saved semantic protos (EMA α={self.proto_ema_alpha}): {path}")
        return str(path)


def weighted_average(metrics):
    """Weighted average for numeric Flower metrics."""
    if not metrics:
        return {}
    skip_keys = {
        "client_id",
        "round",
        "class_phase_counts_json",
        "global_feature_json",
        "prototype_json",
        "prototype_var_json",
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
