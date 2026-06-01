"""Summarize model size and CPU inference cost for paper/deployment reporting.

The script is intentionally independent from training so it can be rerun for
different checkpoints or architectures without touching experiment logic.
"""

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from config import FLConfig
from utils import create_model_by_config, load_shared_weights


def _extract_model_state(obj):
    if isinstance(obj, dict):
        for key in ("model_state", "global_model_state", "model_state_dict", "state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError("Unsupported checkpoint format: no model state dict found")


def _safe_load_state(model, checkpoint_path, device):
    path = Path(checkpoint_path) if checkpoint_path else None
    if not path or not path.exists():
        return False
    ckpt = torch.load(path, map_location=device)
    state = _extract_model_state(ckpt)
    load_shared_weights(model, state, strict=False)
    return True


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    groups = {
        "encoder": 0,
        "classification": 0,
        "regression": 0,
        "other": 0,
    }
    reg_prefixes = (
        "reg_proj.", "reg_transformer.", "reg_attn.", "reg_attn_linear.",
        "reg_heads.", "proto_scale", "proto_bias", "proto_conc",
        "conc_directions", "conc_scale", "conc_bias", "conc_bucket_classifier.",
    )
    cls_prefixes = ("classifier.", "cls_proj.")
    enc_prefixes = (
        "tcn", "tcn_layers", "channel_attn", "self_attn", "attn_linear",
        "feat_proj", "transformer_encoder", "mixstyle",
    )
    for name, param in model.named_parameters():
        n = param.numel()
        if name.startswith(reg_prefixes):
            groups["regression"] += n
        elif name.startswith(cls_prefixes):
            groups["classification"] += n
        elif name.startswith(enc_prefixes):
            groups["encoder"] += n
        else:
            groups["other"] += n
    return total, trainable, groups


def _bytes_to_mib(value):
    return float(value) / (1024.0 * 1024.0)


def _checkpoint_size(path):
    p = Path(path) if path else None
    return p.stat().st_size if p and p.exists() else 0


def _benchmark(fn, warmup=20, repeats=100):
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            fn()
        start = time.perf_counter()
        for _ in range(max(1, repeats)):
            fn()
        elapsed = time.perf_counter() - start
    return elapsed / max(1, repeats)


def _make_reg_config(base_config):
    reg_config = deepcopy(base_config)
    reg_config.USE_REG_LOSS = True
    reg_config.USE_DUAL_PROJ = True
    reg_config.REG_GRAD_DETACH = True
    reg_config.NUM_CONC_BUCKETS = 0
    reg_config.PERSONALIZED_REG = False
    reg_config.SHARE_REG_HEAD = True
    return reg_config


def _summarize_model(name, model, checkpoint_path, device, batch_sizes, repeats, windows_per_file):
    total, trainable, groups = _count_params(model)
    model.eval()
    row = {
        "model": name,
        "checkpoint": str(checkpoint_path) if checkpoint_path else "",
        "checkpoint_size_mib": _bytes_to_mib(_checkpoint_size(checkpoint_path)),
        "total_params": total,
        "trainable_params": trainable,
        "param_memory_mib_fp32": _bytes_to_mib(total * 4),
        "encoder_params": groups["encoder"],
        "classification_params": groups["classification"],
        "regression_params": groups["regression"],
        "other_params": groups["other"],
    }
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 100, 8, device=device)
        if hasattr(model, "forward_reg") and getattr(model, "reg_heads", None) is not None:
            y_cls = torch.zeros(batch_size, dtype=torch.long, device=device)
            y_phase = torch.zeros(batch_size, dtype=torch.long, device=device)

            def run_once():
                _, _, reg_feat = model(x)
                return model.forward_reg(reg_feat, y_cls=y_cls, y_phase=y_phase)
        else:
            def run_once():
                return model(x)[0]

        seconds = _benchmark(run_once, repeats=repeats)
        row[f"latency_ms_batch{batch_size}"] = seconds * 1000.0
        row[f"latency_ms_per_window_batch{batch_size}"] = seconds * 1000.0 / batch_size
        if batch_size == 1:
            row["estimated_ms_per_file_single_window_loop"] = seconds * 1000.0 * windows_per_file
    return row


def _write_markdown(rows, output_path, windows_per_file):
    headers = [
        "model", "total_params", "param_memory_mib_fp32", "checkpoint_size_mib",
        "encoder_params", "classification_params", "regression_params",
        "latency_ms_batch1", "latency_ms_per_window_batch32", "estimated_ms_per_file_single_window_loop",
    ]
    lines = []
    lines.append("# Model Complexity and Deployment Cost Summary")
    lines.append("")
    lines.append(f"Assumed windows per raw measurement file: {windows_per_file}.")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Classification model A is the lightweight routing base used by the pipeline.")
    lines.append("- Regression model B includes the regression branch and per-class regression heads.")
    lines.append("- File-level deployment can batch all windows from one raw measurement, so batch latency is usually more relevant than a single-window loop estimate.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Summarize model complexity and CPU inference cost")
    parser.add_argument("--classifier_checkpoint", default="results/file_fullgrid_cls_A_src45_tgt123/checkpoints/final_model.pth")
    parser.add_argument("--regression_checkpoint", default="results/file_fullgrid_reg_R4_auto_src45_tgt123/checkpoints/separate_regression/separate_regression_source.pth")
    parser.add_argument("--target_regression_checkpoint", default="results/file_fullgrid_reg_R47a_auto_export_src45_tgt123/checkpoints/separate_regression/separate_regression_client1.pth")
    parser.add_argument("--output_dir", default="results/model_complexity_R51")
    parser.add_argument("--batch_sizes", default="1,32")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--windows_per_file", type=int, default=21)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = FLConfig()
    config.DEVICE = str(device)
    config.USE_REG_LOSS = False
    config.USE_DUAL_PROJ = True
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    classifier = create_model_by_config(config, with_reg_head=False).to(device)
    _safe_load_state(classifier, args.classifier_checkpoint, device)

    reg_config = _make_reg_config(config)
    source_reg = create_model_by_config(reg_config, with_reg_head=True).to(device)
    _safe_load_state(source_reg, args.regression_checkpoint, device)

    target_reg = create_model_by_config(reg_config, with_reg_head=True).to(device)
    _safe_load_state(target_reg, args.target_regression_checkpoint, device)

    rows = [
        _summarize_model("classifier_A", classifier, args.classifier_checkpoint, device, batch_sizes, args.repeats, args.windows_per_file),
        _summarize_model("source_regression_B", source_reg, args.regression_checkpoint, device, batch_sizes, args.repeats, args.windows_per_file),
        _summarize_model("target_regression_B_client1", target_reg, args.target_regression_checkpoint, device, batch_sizes, args.repeats, args.windows_per_file),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_complexity.csv"
    json_path = out_dir / "model_complexity.json"
    md_path = out_dir / "model_complexity.md"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(rows, md_path, args.windows_per_file)

    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved summary to {out_dir}")


if __name__ == "__main__":
    main()
