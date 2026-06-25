"""Flower client entrypoint for a local PC or Raspberry Pi edge node."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Optional

import flwr as fl
import torch

from gaps_flower.task import (
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
    ):
        self.client_id = client_id
        self.profile = profile
        self.config = make_config(
            device=device, local_epochs=local_epochs, batch_size=batch_size, profile=profile
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
        logger.info(
            "[GAPS client %d] ready: train_samples=%d, test_samples=%d, device=%s, local_epochs=%d, profile=%s",
            client_id,
            self.train_samples,
            self.test_samples,
            device,
            local_epochs,
            profile,
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
        round_idx = int(config.get("server_round", 1)) if config else 1
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
        if self.config.USE_REPLAY_DISTILL:
            prev_state = self.last_server_state or current_server_state
            set_prev_model_from_state(self.gaps_client, prev_state)
        set_parameters(self.model, parameters, self.parameter_keys)
        self.last_server_state = {
            key: value.detach().cpu().clone()
            for key, value in current_server_state.items()
        }
        arrays, num_examples, metrics = train_one_round(
            self.gaps_client, round_idx, fit_config=config
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
            "replay_distill_enabled": int(bool(self.config.USE_REPLAY_DISTILL)),
        })
        logger.info(
            "[GAPS client %d] fit round=%d DONE: samples=%d, seconds=%.2f",
            self.client_id,
            round_idx,
            num_examples,
            elapsed,
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
    parser.add_argument(
        "--profile",
        choices=("smoke", "gaps_cls", "gaps", "gaps_classification", "classification", "strong_cls"),
        default="smoke",
        help="Client training profile: smoke=CE-only, gaps_cls/strong_cls=CE + server prototype alignment + replay distill",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    client = GapsFlowerClient(
        client_id=args.client_id,
        data_root=args.data_root,
        device=args.device,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        profile=args.profile,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()