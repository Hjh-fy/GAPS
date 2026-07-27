#!/usr/bin/env python3
"""Load a Runtime-v5 package through the real Qt worker and UI wiring."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PyQt5 import QtCore, QtWidgets

from edge_ui_app import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--features", default="")
    parser.add_argument("--reference", default="")
    parser.add_argument("--screenshot", default="")
    parser.add_argument("--maximized", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = Path(args.package_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    screenshot = (
        Path(args.screenshot).expanduser().resolve()
        if args.screenshot
        else None
    )
    if output is not None and output.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {output}")
    if screenshot is not None and screenshot.exists():
        raise FileExistsError(f"REFUSE_TO_OVERWRITE: {screenshot}")
    if bool(args.features) != bool(args.reference):
        raise ValueError("--features and --reference must be provided together")
    replay_window = None
    replay_reference = None
    sensor_fields: list[str] = []
    if args.features:
        features_path = Path(args.features).expanduser().resolve()
        reference_path = Path(args.reference).expanduser().resolve()
        features = np.load(features_path, allow_pickle=False)
        if features.ndim != 3 or features.shape[1:] != (100, 8):
            raise ValueError("features must have shape N×100×8")
        replay_window = np.asarray(features[0], dtype=np.float32)
        with reference_path.open(newline="", encoding="utf-8") as handle:
            replay_reference = next(csv.DictReader(handle))
        manifest = json.loads(
            (package / "manifest.json").read_text(encoding="utf-8")
        )
        sensor_fields = [str(value) for value in manifest["input"]["sensor_fields"]]
    app = QtWidgets.QApplication([])
    started = time.monotonic()
    outcome: dict[str, object] = {
        "status": "FAIL",
        "reason": "timeout",
    }
    state = {"stream_emitted": False}
    with tempfile.TemporaryDirectory(prefix="gaps_edge_ui_qt_smoke_") as temp:
        window = MainWindow(
            data_root=Path(temp),
            default_port="/dev/nonexistent-gaps-smoke",
            default_baudrate=115200,
            max_plot_points=100,
            max_segment_rows=100,
            fullscreen=False,
            font_scale=1.0,
            ai_package=package,
        )
        if args.maximized:
            window.showMaximized()
        else:
            window.show()

        def poll() -> None:
            status = dict(window.last_ai_status)
            if (
                status.get("model_backend") == "gaps_runtime_v5"
                and status.get("runtime_v5_release_id")
                == "gaps_runtime_v5_core_20260726"
                and status.get("runtime_v5_qc_status")
                == "disabled_pending_dependency_audit"
                and status.get("package_fingerprint")
            ):
                if replay_window is not None and not state["stream_emitted"]:
                    state["stream_emitted"] = True
                    base_timestamp = 1_700_000_000.0
                    for sample_index, values in enumerate(replay_window):
                        row = {
                            field: float(values[channel])
                            for channel, field in enumerate(sensor_fields)
                        }
                        row.update(
                            {
                                "timestamp_unix": (
                                    base_timestamp + sample_index / 10.0
                                ),
                                "timestamp_iso": (
                                    f"qt-smoke-sample-{sample_index:03d}"
                                ),
                                "stream_frame_index": sample_index,
                                "connection_id": 1,
                                "frame_plausible": 1,
                                "_recording_active": False,
                                "_recording_session_id": "",
                                "_model_input_precomputed": True,
                            }
                        )
                        window.ai_row_signal.emit(row)
                    return
                if replay_window is not None and not window.last_ai_result:
                    return
                result = dict(window.last_ai_result)
                if replay_reference is not None:
                    differences = {
                        "prediction_ppm": abs(
                            float(result.get("ppm_full_prediction"))
                            - float(replay_reference["prediction_ppm"])
                        ),
                        "source_h1_ppm": abs(
                            float(result.get("ppm_base_prediction"))
                            - float(replay_reference["source_h1_ppm"])
                        ),
                        "max_probability": abs(
                            float(result.get("confidence"))
                            - float(replay_reference["max_probability"])
                        ),
                    }
                    result_ok = (
                        int(result.get("predicted_class"))
                        == int(replay_reference["pred_class"])
                        and result.get("decision")
                        == "disabled_pending_dependency_audit"
                        and result.get("ppm_auto_output") is None
                        and all(value <= 1e-6 for value in differences.values())
                        and "QC disabled; no auto output"
                        in window.ai_result_label.text()
                    )
                    if not result_ok:
                        outcome.update(
                            {
                                "status": "FAIL",
                                "reason": (
                                    "Qt result mapping differs: "
                                    f"result={result}, differences={differences}, "
                                    f"label={window.ai_result_label.text()}"
                                ),
                            }
                        )
                        window.close()
                        app.quit()
                        return
                    outcome["ui_result_verified"] = True
                    outcome["max_abs_differences_first_window"] = differences
                outcome.update(
                    {
                        "status": "PASS",
                        "reason": "",
                        "package_name": status.get("package_name"),
                        "package_fingerprint": status.get(
                            "package_fingerprint"
                        ),
                        "model_backend": status.get("model_backend"),
                        "runtime_v5_release_id": status.get(
                            "runtime_v5_release_id"
                        ),
                        "runtime_v5_qc_status": status.get(
                            "runtime_v5_qc_status"
                        ),
                        "state": status.get("state"),
                    }
                )
                if screenshot is not None:
                    tabs = window.findChild(QtWidgets.QTabWidget, "MainTabs")
                    if tabs is None:
                        outcome.update(
                            {
                                "status": "FAIL",
                                "reason": "MainTabs widget was not found",
                            }
                        )
                    else:
                        tabs.setCurrentIndex(2)
                        app.processEvents()
                        viewport = window.edge_ai_scroll.viewport()
                        horizontal_cards_visible = []
                        for card in window.ai_cards:
                            top_left = card.mapTo(viewport, QtCore.QPoint(0, 0))
                            right = top_left.x() + card.width()
                            horizontal_cards_visible.append(
                                top_left.x() >= -1
                                and right <= viewport.width() + 1
                            )
                        layout_diagnostics = {
                            "compact_mode": bool(window._compact_ui),
                            "window_size": [window.width(), window.height()],
                            "edge_viewport_size": [
                                viewport.width(),
                                viewport.height(),
                            ],
                            "edge_content_size": [
                                window.edge_ai_scroll.widget().width(),
                                window.edge_ai_scroll.widget().height(),
                            ],
                            "all_ai_cards_horizontally_visible": all(
                                horizontal_cards_visible
                            ),
                            "horizontal_scrollbar_policy": int(
                                window.edge_ai_scroll.horizontalScrollBarPolicy()
                            ),
                        }
                        outcome["layout_diagnostics"] = layout_diagnostics
                        if not layout_diagnostics[
                            "all_ai_cards_horizontally_visible"
                        ]:
                            outcome.update(
                                {
                                    "status": "FAIL",
                                    "reason": (
                                        "one or more Edge AI cards overflow "
                                        "the horizontal viewport"
                                    ),
                                }
                            )
                        pixmap = window.grab()
                        screenshot.parent.mkdir(parents=True, exist_ok=True)
                        if not pixmap.save(str(screenshot), "PNG"):
                            outcome.update(
                                {
                                    "status": "FAIL",
                                    "reason": (
                                        f"failed to save screenshot: {screenshot}"
                                    ),
                                }
                            )
                        else:
                            outcome["screenshot"] = str(screenshot)
                            outcome["screenshot_size"] = [
                                pixmap.width(),
                                pixmap.height(),
                            ]
                window.close()
                app.quit()
                return
            if window.ai_state_label.objectName() == "AIErrorPill":
                outcome.update(
                    {
                        "status": "FAIL",
                        "reason": window.ai_result_label.text(),
                    }
                )
                window.close()
                app.quit()
                return
            if time.monotonic() - started >= args.timeout_s:
                outcome.update(
                    {
                        "status": "FAIL",
                        "reason": (
                            "timeout while waiting for Runtime-v5 package; "
                            f"last_status={status}"
                        ),
                    }
                )
                window.close()
                app.quit()

        timer = QtCore.QTimer()
        timer.timeout.connect(poll)
        timer.start(100)
        app.exec_()

    outcome.update(
        {
            "schema_version": "gaps.edge_ui.qt_runtime_v5_smoke.v1",
            "package_dir": str(package),
            "elapsed_seconds_diagnostic": time.monotonic() - started,
        }
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(outcome, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    return 0 if outcome["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
