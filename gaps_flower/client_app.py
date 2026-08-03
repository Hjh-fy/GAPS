"""Flower client entrypoint for a local PC or Raspberry Pi edge node."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import OrderedDict
from typing import Optional

import flwr as fl
import torch

from gaps_flower.observability import NullObserver, load_observer
from gaps_flower.p0i_adaptation import parameter_fingerprint
from gaps_flower.scaffold import (
    ScaffoldClientControlState,
    pack_control_variates,
    unpack_control_variates,
)
from gaps_flower.task import (
    CLASSIFICATION_PROFILE_FLAGS,
    canonical_profile,
    create_model,
    evaluate,
    get_parameters,
    load_client_loaders,
    make_client,
    make_config,
    parameters_to_state_dict,
    set_parameters,
    set_prev_model_from_state,
    train_one_round,
)

logger = logging.getLogger(__name__)


class GapsFlowerClient(fl.client.NumPyClient):
    """GAPS Flower 边缘客户端

    包装原 GAPS Client，提供 Flower NumPyClient 接口。
    每次 fit 调用记录开始/结束时间、样本量、本地轮数到日志。
    """

    def __init__(
        self,
        client_id: int,
        data_root: str,
        device: str,
        local_epochs: int,
        batch_size: int,
        profile: str = "smoke",
        seed: int = 42,
        proximal_mu: float = 0.0,
        optimizer: str = "adam",
        observer=None,
    ):
        self.observer = observer or NullObserver()
        self.client_id = client_id
        self.profile = profile
        self.canonical_profile = canonical_profile(profile)
        self.seed = int(seed)
        self.optimizer = str(optimizer).lower()
        if self.optimizer not in {"adam", "scaffold_sgd"}:
            raise ValueError(f"Unsupported client optimizer: {optimizer}")
        if self.optimizer == "scaffold_sgd" and (
            self.canonical_profile != "ce_only" or float(proximal_mu) != 0.0
        ):
            raise ValueError(
                "FAIL_CLOSED canonical SCAFFOLD requires ce_only and proximal_mu=0"
            )
        self.config = make_config(
            device=device,
            local_epochs=local_epochs,
            batch_size=batch_size,
            profile=profile,
            seed=seed,
            proximal_mu=proximal_mu,
        )
        self.model = create_model(self.config)
        self.parameter_keys = get_parameters(self.model)[1]
        train_loader, test_loader = load_client_loaders(
            data_root, client_id, self.config
        )
        self.gaps_client = make_client(
            client_id, self.model, train_loader, self.config
        )
        self.test_loader = test_loader
        self.train_samples = len(train_loader.dataset)
        self.test_samples = len(test_loader.dataset)
        self.last_server_state: Optional[dict[str, torch.Tensor]] = None
        self.scaffold_state = (
            ScaffoldClientControlState.from_model(self.model)
            if self.optimizer == "scaffold_sgd"
            else None
        )
        logger.info(
            "[GAPS client %d] ready: train_samples=%d, test_samples=%d, device=%s, local_epochs=%d, profile=%s, seed=%d, fedprox_mu=%.6g",
            client_id,
            self.train_samples,
            self.test_samples,
            device,
            local_epochs,
            profile,
            self.seed,
            float(getattr(self.config, "FEDPROX_MU", 0.0)),
        )

    def get_parameters(self, config):
        arrays, _keys = get_parameters(self.model)
        return arrays

    def fit(self, parameters, config):
        """本地训练一轮

        记录:
          - fit开始时间（perf_counter）
          - client_id
          - train_samples（训练集样本数）
          - local_epochs（本地训练轮数）
          - fit_seconds（训练耗时秒数）
          - 以及 train_one_round 返回的 prototype 统计量
        """
        observer = getattr(self, "observer", None) or NullObserver()
        round_idx = int(config.get("server_round", 1)) if config else 1
        fit_callback_start_ns = time.perf_counter_ns()
        observer.emit(
            "client_fit_start",
            round_idx=round_idx,
            client_id=f"C{self.client_id}",
            status="started",
            payload={},
        )
        fit_start = time.perf_counter()
        logger.info(
            "[GAPS client %d] fit round=%d START: train_samples=%d, local_epochs=%d",
            self.client_id,
            round_idx,
            self.train_samples,
            self.config.LOCAL_EPOCHS,
        )

        current_server_state = parameters_to_state_dict(
            parameters, self.parameter_keys, self.model.state_dict()
        )
        received_server_fingerprint = parameter_fingerprint(self.parameter_keys, parameters)
        if self.config.USE_REPLAY_DISTILL:
            if self.last_server_state is not None:
                set_prev_model_from_state(self.gaps_client, self.last_server_state)
            else:
                self.gaps_client.prev_model = None
        set_parameters(self.model, parameters, self.parameter_keys)
        self.last_server_state = {
            key: value.detach().cpu().clone()
            for key, value in current_server_state.items()
        }
        observer.emit(
            "client_train_start",
            round_idx=round_idx,
            client_id=f"C{self.client_id}",
            status="started",
            payload={},
        )
        train_start_ns = time.perf_counter_ns()
        optimizer_name = getattr(self, "optimizer", "adam")
        if optimizer_name == "scaffold_sgd":
            if self.scaffold_state is None:
                raise RuntimeError("FAIL_CLOSED missing persistent SCAFFOLD client state")
            packed_server_control = config.get("scaffold_server_control") if config else None
            if not isinstance(packed_server_control, bytes):
                raise RuntimeError("FAIL_CLOSED missing SCAFFOLD server control payload")
            trainable_reference = OrderedDict(
                (name, parameter.detach().cpu())
                for name, parameter in self.model.named_parameters()
            )
            server_control = unpack_control_variates(
                packed_server_control,
                list(trainable_reference),
                trainable_reference,
            )
            scaffold_result = self.scaffold_state.train(
                self.model,
                self.gaps_client.train_loader,
                server_control=server_control,
                lr=float(config.get("scaffold_lr", 5e-4)),
                local_epochs=self.config.LOCAL_EPOCHS,
                device=torch.device(self.config.DEVICE),
            )
            arrays = [
                scaffold_result.model_state[key].numpy() for key in self.parameter_keys
            ]
            num_examples = int(self.train_samples)
            metrics = {
                "scaffold_control_delta": pack_control_variates(
                    scaffold_result.control_delta
                ),
                "scaffold_local_steps": int(scaffold_result.steps),
                "scaffold_optimizer": scaffold_result.optimizer_name,
                "scaffold_optimizer_lr": float(scaffold_result.optimizer_lr),
                "scaffold_adam_state_present": int(
                    scaffold_result.adam_state_present
                ),
                "scaffold_client_control_before_fingerprint": scaffold_result.client_control_before_fingerprint,
                "scaffold_client_control_after_fingerprint": scaffold_result.client_control_after_fingerprint,
                "scaffold_ce_trajectory_json": json.dumps(
                    scaffold_result.ce_trajectory
                ),
                "scaffold_grad_norms_json": json.dumps(scaffold_result.grad_norms),
                "scaffold_parameter_norms_json": json.dumps(
                    scaffold_result.parameter_norms
                ),
                "train_ce_mean": float(
                    sum(scaffold_result.ce_trajectory)
                    / len(scaffold_result.ce_trajectory)
                ),
                "train_accuracy": float(scaffold_result.train_accuracy),
                "train_metric_examples": int(num_examples),
                "train_ce_averaging": "sample_weighted_over_local_minibatches",
            }
        else:
            arrays, num_examples, metrics = train_one_round(
                self.gaps_client, round_idx, fit_config=config
            )
        train_end_ns = time.perf_counter_ns()
        observer.emit(
            "client_train_end",
            round_idx=round_idx,
            client_id=f"C{self.client_id}",
            status="succeeded",
            payload={"client_train_core_ns": train_end_ns - train_start_ns},
        )
        elapsed = time.perf_counter() - fit_start
        metrics.update({
            "fit_seconds": float(elapsed),
            "round": int(round_idx),
            "client_id": int(self.client_id),
            "num_examples": int(num_examples),
            "local_epochs": int(self.config.LOCAL_EPOCHS),
            "train_samples": int(self.train_samples),
            "profile": self.profile,
            "canonical_profile": self.canonical_profile,
            "seed": self.seed,
            "align_enabled": int(bool(self.config.USE_ALIGN)),
            "replay_distill_enabled": int(bool(self.config.USE_REPLAY_DISTILL)),
            "proto_decoupling_enabled": int(bool(self.config.USE_PROTO_DECOUPLING)),
            "fedprox_mu": float(getattr(self.config, "FEDPROX_MU", 0.0)),
            "optimizer": optimizer_name,
            "server_parameters_fingerprint": received_server_fingerprint,
        })
        logger.info(
            "[GAPS client %d] fit round=%d DONE: samples=%d, seconds=%.2f",
            self.client_id,
            round_idx,
            num_examples,
            elapsed,
        )
        fit_callback_end_ns = time.perf_counter_ns()
        observer.emit(
            "client_fit_end",
            round_idx=round_idx,
            client_id=f"C{self.client_id}",
            status="succeeded",
            payload={
                "client_fit_callback_ns": (
                    fit_callback_end_ns - fit_callback_start_ns
                )
            },
        )
        return arrays, num_examples, metrics

    def evaluate(self, parameters, config):
        """本地评估"""
        start = time.perf_counter()
        current_server_state = parameters_to_state_dict(
            parameters, self.parameter_keys, self.model.state_dict()
        )
        set_parameters(self.model, parameters, self.parameter_keys)
        self.last_server_state = {
            key: value.detach().cpu().clone()
            for key, value in current_server_state.items()
        }
        loss, num_examples, metrics = evaluate(
            self.model, self.test_loader, self.config, client_id=self.client_id
        )
        elapsed = time.perf_counter() - start
        metrics["evaluate_seconds"] = float(elapsed)
        logger.info(
            "[GAPS client %d] evaluate: samples=%d, accuracy=%.4f, seconds=%.2f",
            self.client_id,
            num_examples,
            metrics.get("accuracy", 0.0),
            elapsed,
        )
        return loss, num_examples, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GAPS Flower edge client")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--optimizer",
        choices=("adam", "scaffold_sgd"),
        default="adam",
        help="adam for frozen FedAvg/FedProx/GAPS; scaffold_sgd for canonical SCAFFOLD",
    )
    parser.add_argument(
        "--proximal-mu",
        type=float,
        default=0.0,
        help="FedProx proximal coefficient; 0.0 reproduces the existing local objective",
    )
    parser.add_argument("--observer-context")
    parser.add_argument("--observer-events")
    parser.add_argument(
        "--profile",
        choices=tuple(CLASSIFICATION_PROFILE_FLAGS) + (
            "smoke",
            "gaps_cls",
            "gaps",
            "gaps_classification",
            "classification",
            "strong_cls",
        ),
        default="smoke",
        help="Client training profile: smoke=CE-only, gaps_cls/strong_cls=CE + server prototype alignment + replay distill",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    observer = load_observer(args.observer_context, args.observer_events)
    try:
        client = GapsFlowerClient(
            client_id=args.client_id,
            data_root=args.data_root,
            device=args.device,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            profile=args.profile,
            seed=args.seed,
            proximal_mu=args.proximal_mu,
            optimizer=args.optimizer,
            observer=observer,
        )
        fl.client.start_numpy_client(
            server_address=args.server_address, client=client
        )
    finally:
        observer.close()


if __name__ == "__main__":
    main()
