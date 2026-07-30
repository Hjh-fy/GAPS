#!/usr/bin/env python3
"""Load A4 through the real Qt worker and replay one classification window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from edge_ui_app import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--linger-s", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


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

    features = np.load(features_path).astype(np.float32, copy=False)
    if features.ndim != 3 or features.shape[1:] != (100, 6):
        raise ValueError(f"features must be N×100×6, got {features.shape}")
    norm = np.load(package_dir / "norm_stats.npz")
    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, 6)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, 6)
    unnormalized = (features[0] * std + mean).astype(np.float32)
    manifest = json.loads(
        (package_dir / "manifest.json").read_text(encoding="utf-8")
    )
    sensor_fields = [str(value) for value in manifest["input"]["sensor_fields"]]

    app = QtWidgets.QApplication([])
    started = time.monotonic()
    outcome: dict[str, object] = {"status": "FAIL", "reason": "timeout"}
    state = {
        "phase_sent": False,
        "next_row": 0,
        "finished": False,
        "outcome_ready": False,
    }
    with tempfile.TemporaryDirectory(prefix="gaps_lab_a4_worker_") as temp:
        window = MainWindow(
            data_root=Path(temp),
            default_port="/dev/nonexistent-gaps-lab-a4-smoke",
            default_baudrate=115200,
            max_plot_points=100,
            max_segment_rows=100,
            fullscreen=False,
            font_scale=1.0,
            ai_package=package_dir,
        )
        window.resize(args.width, args.height)
        window.show()

        def finish() -> None:
            if state["finished"]:
                return
            state["finished"] = True
            if screenshot_path is not None:
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                if not window.grab().save(str(screenshot_path)):
                    outcome["status"] = "FAIL"
                    outcome["reason"] = "screenshot_save_failed"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            window.close()
            app.quit()

        def write_outcome() -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        def poll() -> None:
            nonlocal outcome
            if time.monotonic() - started > args.timeout_s:
                outcome = {
                    "status": "FAIL",
                    "reason": "timeout",
                    "last_ai_status": window.last_ai_status,
                    "last_ai_result": window.last_ai_result,
                }
                finish()
                return
            status = dict(window.last_ai_status)
            if (
                status.get("task_type") == "classification"
                and not bool(status.get("has_concentration", True))
                and not state["phase_sent"]
            ):
                window.ai_phase_signal.emit("exposure")
                state["phase_sent"] = True
                return
            if (
                state["phase_sent"]
                and state["next_row"] < len(unnormalized)
            ):
                index = int(state["next_row"])
                values = unnormalized[index]
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
                        for field, value in zip(sensor_fields, values)
                    }
                )
                window.ai_row_signal.emit(row)
                state["next_row"] = index + 1
                return
            result = dict(window.last_ai_result)
            if not result or state["outcome_ready"]:
                return
            app.processEvents()
            content = window.centralWidget()
            checks = {
                "worker_loaded_classification_package": (
                    status.get("task_type") == "classification"
                ),
                "has_no_concentration": (
                    result.get("has_concentration") is False
                    and result.get("ppm_base_prediction") is None
                    and result.get("ppm_full_prediction") is None
                    and result.get("ppm_auto_output") is None
                ),
                "qc_unavailable": (
                    window.ai_qc_card.value.text() == "Unavailable"
                ),
                "consensus_card": (
                    window.ai_ppm_card.title.text() == "Exposure Consensus"
                    and window.ai_ppm_card.unit.text() == "class"
                ),
                "summary_has_no_ppm": (
                    "ppm" not in window.ai_result_label.text().lower()
                ),
                "content_fits_viewport": (
                    content.width() <= window.width()
                    and content.height() <= window.height()
                ),
                "edge_ai_has_no_vertical_overflow": (
                    not window.edge_ai_scroll.verticalScrollBar().isVisible()
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
                "qt_platform": app.platformName(),
                "window_size": [window.width(), window.height()],
                "content_size": [content.width(), content.height()],
                "predicted_gas": result.get("predicted_gas"),
                "confidence": result.get("confidence"),
                "consensus_predicted_gas": result.get(
                    "consensus_predicted_gas"
                ),
                "package_fingerprint": result.get("package_fingerprint"),
                "result_label": window.ai_result_label.text(),
            }
            state["outcome_ready"] = True
            window.tabs.setCurrentIndex(2)
            app.processEvents()
            if args.linger_s > 0:
                write_outcome()
                QtCore.QTimer.singleShot(
                    int(round(args.linger_s * 1000.0)), finish
                )
            else:
                finish()

        timer = QtCore.QTimer()
        timer.timeout.connect(poll)
        timer.start(50)
        app.exec_()

    print(json.dumps(outcome, ensure_ascii=False))
    return 0 if outcome.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
