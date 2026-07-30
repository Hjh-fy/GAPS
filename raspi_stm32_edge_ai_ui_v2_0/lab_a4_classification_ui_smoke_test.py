#!/usr/bin/env python3
"""Validate A4 runtime output and classification UI semantics on the PC.

Torch inference runs synchronously before Qt is created.  This avoids a known
Windows-native heap failure when MultiheadAttention is first loaded inside a
Qt worker thread; the real worker path is reserved for the Raspberry Pi smoke
test where the deployed UI environment is authoritative.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from PyQt5 import QtWidgets

from data_buffer import ADC_FIELD_NAMES
from edge_ai_runtime import EdgeAIRuntime
from edge_ui_app import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--tab",
        choices=("curve", "data", "ai"),
        default="curve",
    )
    return parser.parse_args()


def runtime_result(package_dir: Path, features_path: Path) -> tuple[dict, dict]:
    features = np.load(features_path).astype(np.float32, copy=False)
    if features.ndim != 3 or features.shape[1:] != (100, 6):
        raise ValueError(f"features must be N×100×6, got {features.shape}")
    runtime = EdgeAIRuntime(package_dir)
    mean = np.broadcast_to(runtime.package.mean, (100, 6)).astype(np.float32)
    std = np.broadcast_to(runtime.package.std, (100, 6)).astype(np.float32)
    unnormalized = (features[0] * std + mean).astype(np.float32)
    runtime.set_experiment_phase("exposure")
    result = None
    for index, values in enumerate(unnormalized):
        row = {
            "timestamp_unix": float(index),
            "timestamp_iso": f"replay-{index:03d}",
            "stream_frame_index": index,
            "connection_id": 1,
            "frame_plausible": 1,
            "_model_input_precomputed": True,
            "_recording_active": False,
            "_recording_session_id": "",
        }
        row.update(
            {
                field: float(value)
                for field, value in zip(runtime.package.sensor_fields, values)
            }
        )
        result = runtime.append_row(row)
    if result is None:
        raise RuntimeError("A4 runtime produced no output for a complete window")
    return runtime.status(), result.to_dict()


def seed_visual_sensor_preview(window: MainWindow) -> None:
    """Populate the visual smoke only; production input remains real serial data."""
    for index in range(240):
        row = {
            "elapsed_s": index * 0.25,
            "aht20_rh": 47.2,
            "aht20_temp_c": 24.8,
            "uart4_wz_h3_nk_ppb": 0.0,
            "uart5_tb200b_ppb": 0.0,
        }
        for channel, field in enumerate(ADC_FIELD_NAMES):
            baseline = 1580.0 + channel * 42.0
            response = 210.0 / (1.0 + np.exp(-(index - 115) / 13.0))
            ripple = 8.0 * np.sin(index / 10.0 + channel * 0.25)
            row[field] = baseline + response + ripple
        window.buffer.append_row(row)
    window.refresh_plots()
    window.select_sensor(window.selected_sensor_index)
    window.rh_card.set_value(47.2, "{:.1f}")
    window.temp_card.set_value(24.8, "{:.1f}")
    window.disk_card.set_value(18.6, "{:.1f}")
    window.saved_card.set_value(240, "{}")
    window.segment_card.set_value(240, "{}")
    window.status_summary_label.setText(
        "Visual smoke preview: curve rendering and recording controls are ready."
    )


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    screenshot_path = (
        args.screenshot.expanduser().resolve() if args.screenshot else None
    )
    for path in (output_path, screenshot_path):
        if path is not None and path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")

    status, result = runtime_result(package_dir, features_path)
    app = QtWidgets.QApplication([])
    with tempfile.TemporaryDirectory(prefix="gaps_lab_a4_qt_") as temp:
        window = MainWindow(
            data_root=Path(temp),
            default_port="/dev/nonexistent-gaps-lab-a4-smoke",
            default_baudrate=115200,
            max_plot_points=100,
            max_segment_rows=100,
            fullscreen=False,
            font_scale=1.0,
            ai_package=None,
        )
        window.resize(args.width, args.height)
        window.show()
        window.on_ai_status(status)
        window.on_ai_result(result)
        if args.tab in {"curve", "data"}:
            seed_visual_sensor_preview(window)
        window.tabs.setCurrentIndex({"curve": 0, "data": 1, "ai": 2}[args.tab])
        app.processEvents()
        content = window.centralWidget()
        checks = {
            "task_type_classification": (
                result.get("task_type") == "classification"
            ),
            "has_no_concentration": (
                result.get("has_concentration") is False
                and result.get("ppm_base_prediction") is None
                and result.get("ppm_full_prediction") is None
                and result.get("ppm_auto_output") is None
            ),
            "qc_unavailable": (
                result.get("decision") == "unavailable_qc_not_validated"
                and window.ai_qc_card.value.text() == "Unavailable"
            ),
            "consensus_card": (
                window.ai_ppm_card.title.text() == "Exposure Consensus"
                and window.ai_ppm_card.unit.text() == "class"
            ),
            "summary_has_no_ppm": (
                "ppm" not in window.ai_result_label.text().lower()
            ),
            "probability_vector_has_three_classes": (
                len(result.get("class_probabilities", [])) == 3
            ),
            "window_is_800x480": (
                window.width() == args.width and window.height() == args.height
            ),
            "content_fits_viewport": (
                content.width() <= window.width()
                and content.height() <= window.height()
            ),
            "active_tab_has_no_vertical_overflow": (
                args.tab != "ai"
                or not window.edge_ai_scroll.verticalScrollBar().isVisible()
            ),
            "all_six_cards_horizontally_visible": all(
                card.mapTo(window, card.rect().topLeft()).x() >= 0
                and card.mapTo(window, card.rect().bottomRight()).x()
                <= window.width()
                for card in window.ai_cards
            ),
        }
        outcome = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "window_size": [window.width(), window.height()],
            "content_size": [content.width(), content.height()],
            "predicted_gas": result.get("predicted_gas"),
            "confidence": result.get("confidence"),
            "consensus_predicted_gas": result.get("consensus_predicted_gas"),
            "consensus_window_count": result.get("consensus_window_count"),
            "package_fingerprint": result.get("package_fingerprint"),
            "result_label": window.ai_result_label.text(),
            "pc_worker_boundary": (
                "Windows Qt-worker Torch smoke is not authoritative because "
                "the native runtime raised 0xc0000374; Pi worker validation pending."
            ),
        }
        if screenshot_path is not None:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(screenshot_path)):
                raise RuntimeError(f"failed to save screenshot: {screenshot_path}")
        window.close()
        app.processEvents()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if outcome["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
