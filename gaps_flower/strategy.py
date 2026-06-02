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

            # ── 阶段 4.4: 服务端域适应 ──
            if self.use_domain_adapt and server_round > self.domain_adapt_warmup:
                da_path, da_summary = self._run_domain_adapt(
                    server_round, aggregated_state, arrays
                )
                event["domain_adapt"] = da_path
                event["domain_adapt_summary"] = da_summary

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

    阶段 4.1：语义原型 EMA 聚合
    - 每轮从 client 上传的 class-phase μ_cp / σ²_cp 做加权平均后，
      通过 EMA 更新全局语义原型 semantic_protos

    阶段 4.2：选择性聚合
    - 计算每个 client 原型与 EMA 语义原型的余弦相似度
    - 相似度低的 client 权重被压制（漂移过大）
    - w'_i ∝ max(cos_sim_i, s_min) · n_i

    阶段 4.3：原型级域适应诊断
    - 计算 client-to-EMA 原型漂移 (L2)
    - 计算 client-to-client 原型发散步 (pairwise L2)
    - 纯诊断，不修改模型参数，用于观测域漂移趋势

    阶段 4.4：服务端域适应 (CORAL + MMD + 对抗训练)
    - 聚合后对全局模型做 K 步服务端优化
    - 使用源域验证集 + 目标域校准集对齐特征分布
    - 产出 domain-adapted checkpoint
    """

    def __init__(
        self,
        *,
        proto_ema_alpha: float = 0.8,
        use_selective_agg: bool = True,
        selective_warmup: int = 3,
        selective_min_scale: float = 0.3,
        use_proto_mmd: bool = True,
        use_domain_adapt: bool = False,
        server_val_data: Optional[str] = None,
        server_calib_data: Optional[str] = None,
        domain_adapt_steps: int = 30,
        domain_adapt_warmup: int = 3,
        da_use_coral: bool = True,
        da_use_mmd: bool = True,
        da_use_adversarial: bool = False,
        da_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.proto_ema_alpha = proto_ema_alpha
        self.semantic_protos: Dict[str, torch.Tensor] = {}
        self.semantic_proto_vars: Dict[str, torch.Tensor] = {}
        self.use_selective_agg = use_selective_agg
        self.selective_warmup = selective_warmup
        self.selective_min_scale = selective_min_scale
        self.use_proto_mmd = use_proto_mmd

        self.use_domain_adapt = use_domain_adapt
        self.server_val_data = server_val_data
        self.server_calib_data = server_calib_data
        self.domain_adapt_steps = domain_adapt_steps
        self.domain_adapt_warmup = domain_adapt_warmup
        self.da_use_coral = da_use_coral
        self.da_use_mmd = da_use_mmd
        self.da_use_adversarial = da_use_adversarial
        self.da_device = da_device
        self._da_trainer = None
        self._val_loader = None
        self._calib_loader = None

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

        # ── 2. 计算基础聚合权重 w_i = n_i / Σn_j ──
        total_examples = sum(num_examples_list)
        base_weights = torch.tensor(
            [n / total_examples for n in num_examples_list],
            dtype=torch.float32,
        )

        # ── 阶段 4.2: 选择性聚合权重调整 ──
        selective_info: dict = {"selective_active": False}
        if self.use_selective_agg and self.semantic_protos and server_round > self.selective_warmup:
            selective_weights, selective_info = self._compute_selective_weights(
                results, base_weights, server_round
            )
            weights = selective_weights
        else:
            weights = base_weights
            if self.use_selective_agg:
                selective_info = {
                    "selective_active": False,
                    "reason": f"warmup (round {server_round} ≤ {self.selective_warmup})" if self.semantic_protos else "no semantic_protos yet",
                }

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
        event["selective_agg"] = selective_info
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

            # ── 阶段 4.3: 原型级域适应诊断 (MMD) ──
            if self.use_proto_mmd and results:
                mmd_path, mmd_summary = self._compute_proto_mmd_diagnostics(
                    server_round, results
                )
                event["proto_mmd"] = mmd_path
                event["proto_mmd_summary"] = mmd_summary

        if aggregated_parameters is not None:
            arrays = parameters_to_ndarrays(aggregated_parameters)
            event["checkpoint"] = self._save_checkpoint(server_round, arrays)
        self._write_history()
        return aggregated_parameters, aggregated_metrics

    # ═══════════════════════════════════════════════════════════════
    # 阶段 4.2: 选择性聚合
    # ═══════════════════════════════════════════════════════════════

    def _compute_selective_weights(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        base_weights: torch.Tensor,
        server_round: int,
    ) -> Tuple[torch.Tensor, dict]:
        """计算选择性聚合权重

        数学公式:
          s_i = (1/|K_i|) · Σ_{k∈K_i} cos(μ_{i,k}, μ^{sem}_k)
          w'_i = max(s_i, s_min) · w_i
          w' = w' / Σw'  (重归一化)

        其中 K_i 是客户端 i 上传的所有 (cls,phase) 键集合

        Args:
            results: Flower fit results 列表
            base_weights: 基础权重 (n_i / Σn_j)
            server_round: 当前轮次

        Returns:
            (调整后权重, 诊断信息字典)
        """
        similarities: List[float] = []
        client_ids: List[int] = []
        per_key_details: Dict[str, Dict[str, float]] = {}

        for _proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            cid = int(metrics.get("client_id", -1))
            prototypes = self._parse_json_metric(metrics.get("prototype_json"), {})

            sim_sum = 0.0
            sim_count = 0
            key_sims: Dict[str, float] = {}
            for key, proto_vec in prototypes.items():
                if key in self.semantic_protos:
                    a = torch.tensor(proto_vec, dtype=torch.float32).unsqueeze(0)
                    b = self.semantic_protos[key].unsqueeze(0)
                    cos_sim = float(
                        torch.nn.functional.cosine_similarity(a, b).item()
                    )
                    sim_sum += cos_sim
                    sim_count += 1
                    key_sims[key] = cos_sim

            avg_sim = sim_sum / max(sim_count, 1)
            scaled = max(avg_sim, self.selective_min_scale)
            similarities.append(scaled)
            client_ids.append(cid)
            per_key_details[str(cid)] = {
                "avg_cos_sim": avg_sim,
                "scaled_cos_sim": scaled,
                "n_keys": sim_count,
                "per_key": key_sims,
            }

        sim_tensor = torch.tensor(similarities, dtype=torch.float32)
        adjusted = base_weights * sim_tensor
        adjusted = adjusted / adjusted.sum()

        info = {
            "selective_active": True,
            "round": server_round,
            "warmup": self.selective_warmup,
            "min_scale": self.selective_min_scale,
            "client_similarities": {
                str(cid): float(s) for cid, s in zip(client_ids, similarities)
            },
            "per_key_details": per_key_details,
        }
        return adjusted, info

    # ═══════════════════════════════════════════════════════════════
    # 阶段 4.3: 原型级域适应诊断
    # ═══════════════════════════════════════════════════════════════

    def _compute_proto_mmd_diagnostics(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[str, dict]:
        """计算原型级域漂移诊断指标

        两类指标 (均基于 L2 距离):
          1. client-to-EMA 漂移: 每个 client 的 μ_{i,k} 与 μ^{sem}_k 的 L2 距离
          2. client-to-client 发散: 不同 client 对同一 (cls,phase) 的 pairwise L2

        这些指标纯诊断、不修改模型，用于观测：
          - 哪些 client 漂移最大
          - 哪些 (cls,phase) 跨 client 一致性最差

        Args:
            server_round: 当前轮次
            results: Flower fit results 列表

        Returns:
            (保存路径, 摘要字典)
        """
        # 收集每个客户端按 key 组织的原型
        client_protos: Dict[int, Dict[str, torch.Tensor]] = {}
        for _proxy, fit_res in results:
            metrics = dict(fit_res.metrics or {})
            cid = int(metrics.get("client_id", -1))
            prototypes = self._parse_json_metric(metrics.get("prototype_json"), {})
            client_protos[cid] = {
                key: torch.tensor(vec, dtype=torch.float32)
                for key, vec in prototypes.items()
            }

        # ── 1. client-to-EMA 漂移 ──
        drift_per_client: Dict[str, dict] = {}
        drift_values: List[float] = []
        for cid, protos in sorted(client_protos.items()):
            key_drifts: Dict[str, float] = {}
            for key, proto in protos.items():
                if key in self.semantic_protos:
                    key_drifts[key] = float(
                        torch.norm(proto - self.semantic_protos[key], p=2).item()
                    )
            avg_drift = sum(key_drifts.values()) / max(len(key_drifts), 1)
            drift_per_client[str(cid)] = {
                "avg_l2_drift": avg_drift,
                "n_keys": len(key_drifts),
                "per_key": key_drifts,
            }
            drift_values.append(avg_drift)

        # ── 2. client-to-client pairwise 发散 ──
        all_keys = sorted(
            {key for protos in client_protos.values() for key in protos}
            if client_protos else set()
        )
        key_divergence: Dict[str, dict] = {}
        divergence_values: List[float] = []
        cid_list = sorted(client_protos.keys())
        for key in all_keys:
            vectors = []
            for cid in cid_list:
                if key in client_protos.get(cid, {}):
                    vectors.append(client_protos[cid][key])
            if len(vectors) < 2:
                continue
            pairwise_l2 = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    pairwise_l2.append(
                        float(torch.norm(vectors[i] - vectors[j], p=2).item())
                    )
            avg_div = sum(pairwise_l2) / len(pairwise_l2)
            key_divergence[key] = {
                "avg_pairwise_l2": avg_div,
                "max_pairwise_l2": max(pairwise_l2),
                "n_clients": len(vectors),
            }
            divergence_values.append(avg_div)

        summary = {
            "mean_client_to_ema_drift": float(np.mean(drift_values)) if drift_values else 0.0,
            "max_client_to_ema_drift": float(np.max(drift_values)) if drift_values else 0.0,
            "mean_inter_client_divergence": float(np.mean(divergence_values)) if divergence_values else 0.0,
            "max_inter_client_divergence": float(np.max(divergence_values)) if divergence_values else 0.0,
            "n_drift_clients": len(drift_per_client),
            "n_divergence_keys": len(key_divergence),
        }

        payload = {
            "run_name": self.run_name,
            "round": int(server_round),
            "summary": summary,
            "client_to_ema_drift": drift_per_client,
            "inter_client_divergence": key_divergence,
        }
        path = self.output_dir / f"proto_mmd_round_{server_round:03d}.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest = self.output_dir / "proto_mmd_latest.json"
        latest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[GAPS] Saved proto MMD diagnostics: {path}")
        return str(path), summary

    # ═══════════════════════════════════════════════════════════════
    # 共享工具方法
    # ═══════════════════════════════════════════════════════════════

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

    # ═══════════════════════════════════════════════════════════════
    # 阶段 4.4: 服务端域适应 (CORAL + MMD + 对抗)
    # ═══════════════════════════════════════════════════════════════

    def _run_domain_adapt(
        self, server_round: int, aggregated_state: OrderedDict, arrays: List[np.ndarray]
    ) -> Tuple[str, dict]:
        """在聚合后的全局模型上运行服务端域适应

        Args:
            server_round: 当前联邦轮次
            aggregated_state: 聚合后的 state_dict (OrderedDict)
            arrays: 聚合后的参数数组

        Returns:
            (adapted_checkpoint_path, diagnostics_summary)
        """
        from gaps_flower.domain_adaptation import ServerDomainAdaptation
        import torch

        # ── 1. 懒加载：构建域适应模型 + 数据加载器 ──
        if self._da_trainer is None:
            da_device = torch.device(
                self.da_device if torch.cuda.is_available() else "cpu"
            )
            self._init_domain_adapt_loaders()

            da_model = self._build_da_model(aggregated_state, da_device)
            hyperparams = {
                "USE_DEEP_CORAL": self.da_use_coral,
                "USE_MMD_ALIGNMENT": self.da_use_mmd,
                "USE_ADVERSARIAL_DOMAIN": self.da_use_adversarial,
                "LAMBDA_DEEP_CORAL": 0.1,
                "LAMBDA_GLOBAL_MMD": 0.5,
                "LAMBDA_CLASS_MMD": 0.5,
                "LAMBDA_PROTO_ANCHOR": 0.3,
                "LAMBDA_ADV_DOMAIN": 0.1,
                "SERVER_OPT_LR": 1e-4,
                "HIDDEN_DIM2": 64,
                "NUM_CLASSES": 4,
                "MAX_VAL_BATCHES": 10,
                "ADV_DOMAIN_LR": 0.001,
                "ADV_CRITIC_ITERS": 3,
                "ADV_GRADIENT_PENALTY": 10.0,
                "ADV_CLASS_CONDITIONAL": True,
                "CORAL_CLASS_CONDITIONAL": False,
            }
            self._da_trainer = ServerDomainAdaptation(
                model=da_model,
                val_loader=self._val_loader,
                calib_loader=self._calib_loader,
                semantic_protos=self.semantic_protos,
                device=da_device,
                hyperparams=hyperparams,
            )
        else:
            da_device = self._da_trainer.device
            da_model = self._build_da_model(aggregated_state, da_device)
            self._da_trainer.model = da_model

        # ── 2. 运行域适应优化 ──
        adapted_model, diag = self._da_trainer.run_adaptation(
            num_steps=self.domain_adapt_steps,
        )

        # ── 3. 保存 domain-adapted checkpoint ──
        adapted_state = adapted_model.state_dict()
        da_arrays = []
        for key in self.parameter_keys:
            da_arrays.append(adapted_state[key].detach().cpu().numpy())

        ckpt = {
            "round": int(server_round),
            "model_state": {
                key: value.detach().cpu().clone()
                for key, value in adapted_state.items()
            },
            "parameter_keys": self.parameter_keys,
            "run_name": self.run_name,
            "adaptive": True,
            "diagnostics": diag,
        }
        path = self.output_dir / f"server_round_{server_round:03d}_adapted.pth"
        torch.save(ckpt, path)
        latest = self.output_dir / "server_latest_adapted.pth"
        torch.save(ckpt, latest)
        print(f"[GAPS] Saved domain-adapted checkpoint: {path}")

        # ── 4. 保存诊断文本 ──
        diag_path = self.output_dir / f"domain_adapt_round_{server_round:03d}.json"
        diag_path.write_text(
            json.dumps(diag, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest_diag = self.output_dir / "domain_adapt_latest.json"
        latest_diag.write_text(
            json.dumps(diag, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        return str(path), diag

    def _build_da_model(
        self, aggregated_state: OrderedDict, device: torch.device
    ):
        """根据聚合后的 state_dict 构建域适应模型

        Args:
            aggregated_state: 聚合后的 OrderedDict state
            device: 计算设备

        Returns:
            加载了聚合权重的模型
        """
        from gaps_flower.task import create_model, make_config

        config = make_config(device=str(device), local_epochs=1, batch_size=32)
        model = create_model(config)
        model.load_state_dict(aggregated_state, strict=False)
        return model.to(device)

    def _init_domain_adapt_loaders(self) -> None:
        """初始化服务端数据加载器 (val_loader, calib_loader)

        从 .npy 文件加载校准集和验证集数据。
        """
        from pathlib import Path
        from federated_dataset import GasSensorWindowDataset
        from torch.utils.data import DataLoader
        import numpy as np

        batch_size = 32

        def _load_from_dir(data_dir: str) -> DataLoader:
            dp = Path(data_dir)
            features = np.load(dp / "features.npy")
            cls_labels = np.load(dp / "classification_labels.npy")
            phase_path = dp / "phase_labels.npy"
            phase_labels = (
                np.load(phase_path, allow_pickle=True)
                if phase_path.exists() else None
            )
            dataset = GasSensorWindowDataset(
                features=features,
                regression_labels=np.zeros((len(features), 4), dtype=np.float32),
                classification_labels=cls_labels,
                phase_labels=phase_labels,
                normalize=False,
                mean_std=None,
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )

        if self.server_val_data:
            self._val_loader = _load_from_dir(self.server_val_data)
            print(f"[GAPS] Loaded server val data: {self.server_val_data} "
                  f"({len(self._val_loader.dataset)} samples)")
        else:
            self._val_loader = None

        if self.server_calib_data:
            self._calib_loader = _load_from_dir(self.server_calib_data)
            print(f"[GAPS] Loaded server calib data: {self.server_calib_data} "
                  f"({len(self._calib_loader.dataset)} samples)")
        else:
            self._calib_loader = None


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
