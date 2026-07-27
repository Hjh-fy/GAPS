#!/usr/bin/env python3
"""Raspberry Pi acquisition and auditable edge-AI UI for the STM32 gas-sensor board.

v2.2 combines guarded long-running acquisition with a responsive,
profile-driven GAPS edge-AI runtime:
1. Connect to the real HC-04 Bluetooth serial or USB serial source
2. Parse the current 43-byte / 20-field STM32 frame
3. Select one of 16 sensors with large direct buttons and view its curve alone
4. Show basic receiving status and RH/Temp
5. Create an experiment folder under a configurable save root
6. Save either a continuous raw.csv or the current observed segment
7. Confirm before resetting the observed segment to prevent accidental data loss
8. Show free disk space and stop recording safely if a CSV write fails

No simulated waveform mode is included. If the curve moves, it comes from
parsed real frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from csv_writer import ExperimentSession
from data_buffer import ADC_FIELD_NAMES, SensorRingBuffer
from frame_parser_v20 import CSV_COLUMNS_WITH_DERIVED
from serial_worker import SerialWorker
from edge_ai_worker import EdgeAIWorker
from config_loader import load_config, ui_defaults


DEFAULT_HC04_PORT = "/dev/rfcomm0"
DEFAULT_DATA_ROOT = Path.home() / "GAPS_data" / "experiments"
DEFAULT_BAUDRATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]


class ExperimentDialog(QtWidgets.QDialog):
    """Collect experiment metadata before saving.

    These fields are used for two purposes:
    - Create a readable experiment folder name.
    - Write meta.json for later data review.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New experiment")
        self.setMinimumWidth(420)

        now_name = f"exp_{datetime.now().strftime('%H%M%S')}"
        self.experiment_name = QtWidgets.QLineEdit(now_name)
        self.gas_type = QtWidgets.QComboBox()
        self.gas_type.addItems(["unknown", "air", "Ethanol", "CO", "Ethylene", "Methane", "mixed"])
        self.target_ppm = QtWidgets.QLineEdit("")
        self.repeat_id = QtWidgets.QLineEdit("R1")
        self.operator = QtWidgets.QLineEdit("")
        self.device_id = QtWidgets.QLineEdit("raspi_edge_01")
        self.stm32_board_id = QtWidgets.QLineEdit("stm32_vsensor_01")
        self.note = QtWidgets.QPlainTextEdit()
        self.note.setPlaceholderText("Note: exposure/recovery time, flow setting, sensor condition...")
        self.note.setFixedHeight(70)

        form = QtWidgets.QFormLayout()
        form.addRow("Experiment name", self.experiment_name)
        form.addRow("Gas", self.gas_type)
        form.addRow("Target ppm", self.target_ppm)
        form.addRow("Repeat", self.repeat_id)
        form.addRow("Operator", self.operator)
        form.addRow("Pi node", self.device_id)
        form.addRow("STM32 board", self.stm32_board_id)
        form.addRow("Note", self.note)

        hint = QtWidgets.QLabel(
            "Folder name uses time + experiment name. These fields are metadata for review, not required for receiving data."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintText")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def metadata(self) -> Dict[str, object]:
        return {
            "experiment_name": self.experiment_name.text().strip(),
            "gas_type": self.gas_type.currentText(),
            "target_ppm": self.target_ppm.text().strip(),
            "repeat_id": self.repeat_id.text().strip(),
            "operator": self.operator.text().strip(),
            "raspi_device_id": self.device_id.text().strip(),
            "stm32_board_id": self.stm32_board_id.text().strip(),
            "note": self.note.toPlainText().strip(),
            "ui_version": "raspi_stm32_edge_ai_ui_v2_2",
            "protocol": "0x80 0x81 + 20 uint16 little-endian + 0x82",
        }


class StatCard(QtWidgets.QFrame):
    def __init__(self, title: str, unit: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("StatCard")
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("CardTitle")
        self.value = QtWidgets.QLabel("--")
        self.value.setObjectName("CardValue")
        self.value.setMinimumWidth(0)
        self.value.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.unit = QtWidgets.QLabel(unit)
        self.unit.setObjectName("CardUnit")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(1)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.unit)

    def set_value(self, value: object, fmt: str = "{}") -> None:
        try:
            text = fmt.format(value)
        except Exception:
            text = str(value)
        self.value.setText(text)
        self.value.setToolTip(text)

    def set_status(self, status: str) -> None:
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)


class MainWindow(QtWidgets.QMainWindow):
    ai_row_signal = QtCore.pyqtSignal(dict)
    ai_load_signal = QtCore.pyqtSignal(str)
    ai_unload_signal = QtCore.pyqtSignal()
    ai_reset_signal = QtCore.pyqtSignal(bool)
    ai_phase_signal = QtCore.pyqtSignal(str)
    def __init__(
        self,
        data_root: Path,
        default_port: str,
        default_baudrate: int,
        max_plot_points: int,
        max_segment_rows: int,
        fullscreen: bool = False,
        font_scale: float = 1.12,
        ai_package: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("GAPS Edge AI Monitor")
        self.resize(800, 480)
        self.setMinimumSize(640, 360)
        self.fullscreen_requested = bool(fullscreen)
        self.font_scale = max(0.75, min(float(font_scale), 1.50))

        self.data_root = Path(data_root).expanduser().resolve()
        self.default_port = default_port
        self.default_baudrate = int(default_baudrate)
        self.buffer = SensorRingBuffer(max_points=max_plot_points)
        self.segment_rows: List[Dict[str, object]] = []
        self.max_segment_rows = int(max_segment_rows)
        self.worker: Optional[SerialWorker] = None
        self.session: Optional[ExperimentSession] = None
        self.saving_enabled = False
        self.last_stats: Dict[str, object] = {}
        self.last_frame_wall_time: Optional[float] = None
        self.connection_active = False
        self.last_error_message = ""
        self.selected_sensor_index = 0
        self.sensor_buttons: List[QtWidgets.QPushButton] = []
        self._last_disk_check = 0.0
        self._low_disk_warned = False
        self._stale_logged = False
        self._disconnect_requested_by_user = False
        self._connection_counter = 0
        self._stream_start_timestamp: Optional[float] = None
        self._stream_frame_index = 0
        self._segment_truncated_rows = 0
        self._last_implausible_event_time = 0.0
        self._closing = False
        self.ai_package_path: Optional[Path] = Path(ai_package).expanduser().resolve() if ai_package else None
        self.ai_thread: Optional[QtCore.QThread] = None
        self.ai_worker: Optional[EdgeAIWorker] = None
        self.last_ai_result: Dict[str, object] = {}
        self.last_ai_status: Dict[str, object] = {}
        self._last_ai_fingerprint_logged = ""
        self._last_ai_unsaved_reason = ""
        self._recording_error_latched = False
        self._compact_ui: Optional[bool] = None
        self._ai_history_ppm: List[float] = []
        self._ai_history_index: List[int] = []

        self._build_ui()
        self._apply_theme()
        self._setup_timer()
        self._setup_ai_worker()
        self.refresh_ports()
        self._set_connection_state(False)
        self._set_stream_state("IDLE")
        self.select_sensor(0)
        self.update_disk_space()
        if self.ai_package_path is not None:
            self.ai_load_signal.emit(str(self.ai_package_path))

    # ------------------------- UI construction -------------------------
    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QVBoxLayout(central)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)
        self.top_bar = self._build_top_bar()
        self.main_tabs = self._build_main_tabs()
        self.bottom_bar = self._build_bottom_bar()
        self.main_layout.addWidget(self.top_bar)
        self.main_layout.addWidget(self.main_tabs, stretch=1)
        self.main_layout.addWidget(self.bottom_bar)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_top_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("TopBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(5)

        self.app_title_label = QtWidgets.QLabel("STM32 Monitor")
        self.app_title_label.setObjectName("AppTitle")
        layout.addWidget(self.app_title_label)

        self.stream_state_label = QtWidgets.QLabel("IDLE")
        self.stream_state_label.setObjectName("IdlePill")
        layout.addWidget(self.stream_state_label)

        self.connect_btn = QtWidgets.QPushButton("Connect HC-04")
        self.connect_btn.clicked.connect(self.toggle_connection)
        layout.addWidget(self.connect_btn)

        self.connection_label = QtWidgets.QLabel("Disconnected")
        self.connection_label.setObjectName("DisconnectedLabel")
        layout.addWidget(self.connection_label)

        self.data_age_label = QtWidgets.QLabel("No data")
        self.data_age_label.setObjectName("DataAgeLabel")
        layout.addWidget(self.data_age_label)

        self.quick_good_label = QtWidgets.QLabel("G:0")
        self.quick_bad_label = QtWidgets.QLabel("B:0")
        self.quick_fps_label = QtWidgets.QLabel("0.00 fps")
        for label in [self.quick_good_label, self.quick_bad_label, self.quick_fps_label]:
            label.setObjectName("QuickStat")
            layout.addWidget(label)

        layout.addStretch(1)
        self.port_combo = QtWidgets.QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.setMinimumWidth(170)
        self.port_combo.setToolTip("Real input port, e.g. /dev/rfcomm0, /dev/ttyUSB0 or /dev/ttyACM0")

        self.refresh_btn = QtWidgets.QPushButton("↻")
        self.refresh_btn.setToolTip("Refresh serial ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)

        self.baud_combo = QtWidgets.QComboBox()
        self.baud_combo.setEditable(True)
        baud_values = list(DEFAULT_BAUDRATES)
        if self.default_baudrate not in baud_values:
            baud_values.append(self.default_baudrate)
        self.baud_combo.addItems([str(v) for v in sorted(set(baud_values))])
        self.baud_combo.setCurrentText(str(self.default_baudrate))
        self.baud_combo.setMinimumWidth(75)
        return bar

    def _build_main_tabs(self) -> QtWidgets.QWidget:
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("MainTabs")
        tabs.tabBar().setUsesScrollButtons(False)
        tabs.tabBar().setElideMode(QtCore.Qt.ElideRight)
        # Leading padding avoids a Qt/Wayland first-tab glyph clipping quirk.
        tabs.addTab(self._build_sensor_tab(), "  Live Curve")
        self.status_scroll = self._wrap_scroll(self._build_status_tab())
        self.edge_ai_scroll = self._wrap_scroll(self._build_edge_ai_tab())
        tabs.addTab(self.status_scroll, "Data / Save")
        tabs.addTab(self.edge_ai_scroll, "Edge AI")
        return tabs

    def _wrap_scroll(self, widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def _build_sensor_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QFrame()
        tab.setObjectName("PlotPanel")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(5)

        select_box = QtWidgets.QFrame()
        select_box.setObjectName("SensorSelectBox")
        grid = QtWidgets.QGridLayout(select_box)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setSpacing(4)
        label = QtWidgets.QLabel("Select sensor")
        label.setObjectName("SensorSelectTitle")
        grid.addWidget(label, 0, 0, 1, 2)
        for idx in range(16):
            btn = QtWidgets.QPushButton(str(idx + 1))
            btn.setObjectName("SensorButton")
            btn.setCheckable(True)
            btn.setMinimumHeight(34)
            btn.clicked.connect(lambda checked=False, i=idx: self.select_sensor(i))
            self.sensor_buttons.append(btn)
            row = 1 + idx // 8
            col = idx % 8
            grid.addWidget(btn, row, col)
        self.sensor_value_label = QtWidgets.QLabel("Sensor 1  Latest: --")
        self.sensor_value_label.setObjectName("SensorValueLabel")
        grid.addWidget(self.sensor_value_label, 3, 0, 1, 8)
        layout.addWidget(select_box)

        self.sensor_plot = pg.PlotWidget()
        self.sensor_plot.setBackground(None)
        self.sensor_plot.showGrid(x=True, y=True, alpha=0.22)
        self.sensor_plot.setLabel("bottom", "t", units="s")
        self.sensor_plot.setLabel("left", "ADC raw")
        self.sensor_plot.setTitle("Sensor 1")
        self.sensor_curve = self.sensor_plot.plot([], [], pen=pg.mkPen(pg.intColor(0, hues=16), width=2.6))
        layout.addWidget(self.sensor_plot, stretch=1)
        return tab

    def _build_status_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("SidePanel")
        root = QtWidgets.QVBoxLayout(panel)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        connection_box = QtWidgets.QGroupBox("Input connection")
        connection_layout = QtWidgets.QHBoxLayout(connection_box)
        connection_layout.addWidget(QtWidgets.QLabel("Port"))
        connection_layout.addWidget(self.port_combo, stretch=1)
        connection_layout.addWidget(self.refresh_btn)
        connection_layout.addWidget(QtWidgets.QLabel("Baud"))
        connection_layout.addWidget(self.baud_combo)
        root.addWidget(connection_box)

        cards = QtWidgets.QGridLayout()
        cards.setSpacing(6)
        self.rh_card = StatCard("RH", "%")
        self.temp_card = StatCard("Temp", "°C")
        self.disk_card = StatCard("Disk Free", "GB")
        self.saved_card = StatCard("raw.csv", "rows")
        self.segment_card = StatCard("Segment", "rows")
        self.selected_card = StatCard("Sensor", "selected")
        cards.addWidget(self.rh_card, 0, 0)
        cards.addWidget(self.temp_card, 0, 1)
        cards.addWidget(self.disk_card, 0, 2)
        cards.addWidget(self.saved_card, 1, 0)
        cards.addWidget(self.segment_card, 1, 1)
        cards.addWidget(self.selected_card, 1, 2)
        root.addLayout(cards)

        self.status_summary_label = QtWidgets.QLabel("Not connected. No real data is displayed.")
        self.status_summary_label.setObjectName("StatusSummary")
        self.status_summary_label.setWordWrap(True)
        root.addWidget(self.status_summary_label)

        stats_box = QtWidgets.QGroupBox("Receiving status")
        stats_layout = QtWidgets.QGridLayout(stats_box)
        self.good_label = QtWidgets.QLabel("0")
        self.bad_label = QtWidgets.QLabel("0")
        self.fps_label = QtWidgets.QLabel("0.000")
        self.drop_label = QtWidgets.QLabel("0")
        self.resync_label = QtWidgets.QLabel("0")
        self.buffered_label = QtWidgets.QLabel("0")
        self.implausible_label = QtWidgets.QLabel("0")
        self.aux_label = QtWidgets.QLabel("UART4/UART5 raw: --/--")
        rows = [
            ("Good", self.good_label), ("Bad", self.bad_label), ("FPS", self.fps_label),
            ("Dropped", self.drop_label), ("Resync", self.resync_label),
            ("Buffered", self.buffered_label), ("Implausible", self.implausible_label),
        ]
        for idx, (name, label) in enumerate(rows):
            r, c = divmod(idx, 4)
            stats_layout.addWidget(QtWidgets.QLabel(name), r * 2, c)
            stats_layout.addWidget(label, r * 2 + 1, c)
        stats_layout.addWidget(self.aux_label, 4, 0, 1, 4)
        root.addWidget(stats_box)

        self.save_state_label = QtWidgets.QLabel("Recording: off")
        self.save_state_label.setObjectName("SaveStateLabel")
        self.exp_dir_label = QtWidgets.QLabel("Experiment: --")
        self.exp_dir_label.setWordWrap(True)
        root.addWidget(self.save_state_label)
        root.addWidget(self.exp_dir_label)

        marker_box = QtWidgets.QGroupBox("Experiment event markers")
        marker_layout = QtWidgets.QHBoxLayout(marker_box)
        baseline_btn = QtWidgets.QPushButton("Baseline Start")
        exposure_btn = QtWidgets.QPushButton("Exposure Start")
        recovery_btn = QtWidgets.QPushButton("Recovery Start")
        note_btn = QtWidgets.QPushButton("Add Note")
        baseline_btn.clicked.connect(lambda: self.mark_experiment_event("baseline_start", "Baseline started"))
        exposure_btn.clicked.connect(lambda: self.mark_experiment_event("exposure_start", "Gas exposure started"))
        recovery_btn.clicked.connect(lambda: self.mark_experiment_event("recovery_start", "Recovery started"))
        note_btn.clicked.connect(self.add_experiment_note)
        for btn in [baseline_btn, exposure_btn, recovery_btn, note_btn]:
            marker_layout.addWidget(btn)
        root.addWidget(marker_box)

        log_box = QtWidgets.QGroupBox("Event / error log")
        log_layout = QtWidgets.QVBoxLayout(log_box)
        self.event_log_view = QtWidgets.QPlainTextEdit()
        self.event_log_view.setReadOnly(True)
        self.event_log_view.setMaximumBlockCount(140)
        self.event_log_view.setFixedHeight(100)
        self.event_log_view.setPlaceholderText("Connection, frame and saving messages appear here.")
        log_layout.addWidget(self.event_log_view)
        root.addWidget(log_box)
        return panel

    def _build_edge_ai_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setObjectName("SidePanel")
        root = QtWidgets.QVBoxLayout(panel)
        root.setContentsMargins(7, 7, 7, 7)
        root.setSpacing(7)

        header = QtWidgets.QHBoxLayout()
        self.ai_state_label = QtWidgets.QLabel("AI OFFLINE")
        self.ai_state_label.setObjectName("AIIdlePill")
        header.addWidget(self.ai_state_label)
        self.ai_package_label = QtWidgets.QLabel("No model loaded")
        self.ai_package_label.setObjectName("HintText")
        self.ai_package_label.setMinimumWidth(0)
        self.ai_package_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        header.addWidget(self.ai_package_label, stretch=1)
        self.ai_load_btn = QtWidgets.QPushButton("Load Model")
        self.ai_load_btn.setObjectName("PrimaryButton")
        self.ai_load_btn.clicked.connect(self.choose_ai_package)
        header.addWidget(self.ai_load_btn)
        self.ai_unload_btn = QtWidgets.QPushButton("Unload")
        self.ai_unload_btn.setObjectName("SecondaryButton")
        self.ai_unload_btn.clicked.connect(lambda: self.ai_unload_signal.emit())
        header.addWidget(self.ai_unload_btn)
        self.ai_compact_reset_btn = QtWidgets.QPushButton("Reset")
        self.ai_compact_reset_btn.setToolTip(
            "Keep the baseline and clear the current inference window."
        )
        self.ai_compact_reset_btn.clicked.connect(
            lambda: self.ai_reset_signal.emit(True)
        )
        self.ai_compact_reset_btn.hide()
        header.addWidget(self.ai_compact_reset_btn)
        root.addLayout(header)

        self.ai_cards_layout = QtWidgets.QGridLayout()
        self.ai_cards_layout.setSpacing(6)
        self.ai_gas_card = StatCard("Gas", "class")
        self.ai_ppm_card = StatCard("Concentration", "ppm")
        self.ai_conf_card = StatCard("Confidence", "%")
        self.ai_qc_card = StatCard("QC", "decision")
        self.ai_latency_card = StatCard("Latency", "ms")
        self.ai_rate_card = StatCard("Input Rate", "Hz")
        self.ai_cards = [
            self.ai_gas_card,
            self.ai_ppm_card,
            self.ai_conf_card,
            self.ai_qc_card,
            self.ai_latency_card,
            self.ai_rate_card,
        ]
        self._arrange_ai_cards(3)
        root.addLayout(self.ai_cards_layout)

        self.ai_result_label = QtWidgets.QLabel("Waiting for the first inference window.")
        self.ai_result_label.setWordWrap(True)
        self.ai_result_label.setObjectName("InferenceSummary")
        root.addWidget(self.ai_result_label)

        self.ai_status_grid = QtWidgets.QGridLayout()
        self.ai_status_grid.setHorizontalSpacing(6)
        self.ai_status_grid.setVerticalSpacing(4)
        self.ai_phase_badge = QtWidgets.QLabel("Phase: --")
        self.ai_baseline_badge = QtWidgets.QLabel("Baseline: --")
        self.ai_window_badge = QtWidgets.QLabel("Window: 0/0")
        self.ai_rate_badge = QtWidgets.QLabel("Rate: --")
        self.ai_status_badges = [
            self.ai_phase_badge,
            self.ai_baseline_badge,
            self.ai_window_badge,
            self.ai_rate_badge,
        ]
        for index, badge in enumerate(self.ai_status_badges):
            badge.setObjectName("InfoBadge")
            badge.setAlignment(QtCore.Qt.AlignCenter)
            self.ai_status_grid.addWidget(badge, 0, index)
        root.addLayout(self.ai_status_grid)

        self.ai_window_progress = QtWidgets.QProgressBar()
        self.ai_window_progress.setRange(0, 100)
        self.ai_window_progress.setValue(0)
        self.ai_window_progress.setFormat("Inference window: %v/%m")
        self.ai_window_progress.setTextVisible(True)
        root.addWidget(self.ai_window_progress)

        self.ai_progress_label = QtWidgets.QLabel(
            "Load a verified model package to begin edge inference."
        )
        self.ai_progress_label.setWordWrap(True)
        self.ai_progress_label.setObjectName("HintText")
        root.addWidget(self.ai_progress_label)

        self.ai_history_panel = QtWidgets.QFrame()
        self.ai_history_panel.setObjectName("ChartPanel")
        history_layout = QtWidgets.QVBoxLayout(self.ai_history_panel)
        history_layout.setContentsMargins(7, 5, 7, 7)
        history_layout.setSpacing(3)
        history_title = QtWidgets.QLabel("Recent concentration predictions")
        history_title.setObjectName("SectionTitle")
        history_layout.addWidget(history_title)
        self.ai_history_plot = pg.PlotWidget()
        self.ai_history_plot.setBackground(None)
        self.ai_history_plot.showGrid(x=True, y=True, alpha=0.18)
        self.ai_history_plot.setLabel("bottom", "Inference window")
        self.ai_history_plot.setLabel("left", "Prediction", units="ppm")
        self.ai_history_plot.setMenuEnabled(False)
        self.ai_history_curve = self.ai_history_plot.plot(
            [],
            [],
            pen=pg.mkPen("#38BDF8", width=2.4),
            symbol="o",
            symbolSize=5,
            symbolBrush="#38BDF8",
        )
        history_layout.addWidget(self.ai_history_plot, stretch=1)
        root.addWidget(self.ai_history_panel, stretch=1)

        self.ai_controls_widget = QtWidgets.QWidget()
        controls = QtWidgets.QHBoxLayout(self.ai_controls_widget)
        controls.setContentsMargins(0, 0, 0, 0)
        self.ai_reset_all_btn = QtWidgets.QPushButton("Reset All")
        self.ai_reset_all_btn.setToolTip("Reset the baseline and current inference window.")
        self.ai_reset_all_btn.clicked.connect(lambda: self.ai_reset_signal.emit(False))
        controls.addWidget(self.ai_reset_all_btn)
        self.ai_reset_window_btn = QtWidgets.QPushButton("Reset Window")
        self.ai_reset_window_btn.setToolTip("Keep the baseline and clear only the current inference window.")
        self.ai_reset_window_btn.clicked.connect(lambda: self.ai_reset_signal.emit(True))
        controls.addWidget(self.ai_reset_window_btn)
        controls.addStretch(1)
        root.addWidget(self.ai_controls_widget)

        self.ai_safety_note = QtWidgets.QLabel(
            "Safety: QC-disabled or rejected windows remain auditable, but never "
            "produce an automatic concentration output."
        )
        self.ai_safety_note.setWordWrap(True)
        self.ai_safety_note.setObjectName("SafetyNote")
        root.addWidget(self.ai_safety_note)
        return panel

    def _arrange_ai_cards(self, columns: int) -> None:
        if not hasattr(self, "ai_cards_layout"):
            return
        for card in getattr(self, "ai_cards", []):
            self.ai_cards_layout.removeWidget(card)
        for index, card in enumerate(getattr(self, "ai_cards", [])):
            self.ai_cards_layout.addWidget(
                card,
                index // columns,
                index % columns,
            )
        for column in range(columns):
            self.ai_cards_layout.setColumnStretch(column, 1)

    def _arrange_ai_status_badges(self, columns: int) -> None:
        if not hasattr(self, "ai_status_grid"):
            return
        for badge in getattr(self, "ai_status_badges", []):
            self.ai_status_grid.removeWidget(badge)
        for index, badge in enumerate(getattr(self, "ai_status_badges", [])):
            self.ai_status_grid.addWidget(
                badge,
                index // columns,
                index % columns,
            )
        for column in range(columns):
            self.ai_status_grid.setColumnStretch(column, 1)

    def _set_ai_state_visual(self, text: str, object_name: str) -> None:
        self.ai_state_label.setText(text)
        self.ai_state_label.setObjectName(object_name)
        self.ai_state_label.style().unpolish(self.ai_state_label)
        self.ai_state_label.style().polish(self.ai_state_label)

    @staticmethod
    def _friendly_model_name(package_name: str) -> str:
        name = str(package_name).strip()
        low = name.lower()
        if "runtime_v5" in low and "public" in low:
            return "Public dataset · 10 Hz · Runtime v5"
        if "runtime_v5" in low:
            return "GAPS model · Runtime v5"
        return name or "Verified model package"

    @staticmethod
    def _friendly_qc_decision(decision: str) -> tuple[str, str]:
        normalized = str(decision).strip().lower()
        mapping = {
            "disabled_pending_dependency_audit": ("Unavailable", "warning"),
            "accept": ("Accepted", "ok"),
            "accepted": ("Accepted", "ok"),
            "auto": ("Automatic", "ok"),
            "review": ("Review", "warning"),
            "manual_review": ("Review", "warning"),
            "reject": ("Rejected", "error"),
            "rejected": ("Rejected", "error"),
        }
        return mapping.get(normalized, (str(decision) or "Unknown", "warning"))

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "ai_history_panel"):
            return
        compact = self.width() < 960 or self.height() < 650
        narrow = self.width() < 720
        if self._compact_ui == compact and getattr(self, "_narrow_ui", None) == narrow:
            return
        self._compact_ui = compact
        self._narrow_ui = narrow

        self.app_title_label.setText("GAPS AI" if compact else "STM32 Monitor")
        for widget in [
            self.quick_good_label,
            self.quick_bad_label,
            self.quick_fps_label,
        ]:
            widget.setVisible(not compact)
        self.path_btn.setVisible(not compact)
        self.save_segment_btn.setVisible(not compact)
        self.clear_btn.setVisible(not compact)
        self.sample_label.setVisible(not compact)
        self.open_dir_btn.setVisible(not narrow)
        self.ai_unload_btn.setVisible(not compact)
        self.ai_compact_reset_btn.setVisible(compact)
        self.ai_history_panel.setVisible(not compact)
        self.ai_safety_note.setVisible(not compact)
        self._arrange_ai_cards(2 if narrow else 3)
        self._arrange_ai_status_badges(2 if narrow else 4)
        self._update_ai_detail_visibility()
        self._set_connection_state(self.connection_active)

        if compact:
            self.main_layout.setContentsMargins(4, 4, 4, 4)
            self.main_layout.setSpacing(4)
        else:
            self.main_layout.setContentsMargins(7, 7, 7, 7)
            self.main_layout.setSpacing(7)

    def _update_ai_detail_visibility(self) -> None:
        state = str(self.last_ai_status.get("state", ""))
        compact_ready = bool(self._compact_ui) and state == "ready"
        self.ai_window_progress.setVisible(not compact_ready)
        self.ai_progress_label.setVisible(not compact_ready)
        self.ai_controls_widget.setVisible(not compact_ready)

    def _build_bottom_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QFrame()
        bar.setObjectName("BottomBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(5)

        self.path_btn = QtWidgets.QPushButton("Save Path")
        self.path_btn.setToolTip("Choose the default root folder for new experiment directories.")
        self.path_btn.clicked.connect(self.choose_save_path)
        layout.addWidget(self.path_btn)

        self.new_exp_btn = QtWidgets.QPushButton("New Experiment")
        self.new_exp_btn.clicked.connect(self.create_experiment)
        layout.addWidget(self.new_exp_btn)

        self.save_btn = QtWidgets.QPushButton("Start Recording")
        self.save_btn.setObjectName("PrimaryButton")
        self.save_btn.clicked.connect(self.toggle_saving)
        layout.addWidget(self.save_btn)

        self.save_segment_btn = QtWidgets.QPushButton("Save Segment")
        self.save_segment_btn.setToolTip("Save all frames collected since the last Mark Segment Start.")
        self.save_segment_btn.clicked.connect(self.save_current_segment)
        layout.addWidget(self.save_segment_btn)

        self.clear_btn = QtWidgets.QPushButton("Mark Segment Start")
        self.clear_btn.setToolTip("Reset the displayed curve and observed segment start point. Saved files are not deleted.")
        self.clear_btn.clicked.connect(self.clear_curves)
        layout.addWidget(self.clear_btn)

        self.open_dir_btn = QtWidgets.QPushButton("Data Folder")
        self.open_dir_btn.clicked.connect(self.open_data_folder)
        layout.addWidget(self.open_dir_btn)

        layout.addStretch(1)
        self.sample_label = QtWidgets.QLabel("Latest: --")
        self.sample_label.setMinimumWidth(120)
        layout.addWidget(self.sample_label)
        return bar

    def _apply_theme(self) -> None:
        pg.setConfigOptions(antialias=True)
        base = int(15 * self.font_scale)
        title = int(18 * self.font_scale)
        card_value = int(31 * self.font_scale)
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{ background: #111827; color: #E5E7EB; font-size: {base}px; }}
            QFrame#TopBar, QFrame#BottomBar, QFrame#SidePanel, QFrame#PlotPanel, QFrame#SensorSelectBox {{ background: #1F2937; border-radius: 7px; }}
            QTabWidget#MainTabs::pane {{ border: 1px solid #374151; background: #1F2937; border-radius: 7px; }}
            QTabBar::tab {{ background: #374151; color: #F9FAFB; padding: 6px 14px; margin-right: 2px; border-top-left-radius: 5px; border-top-right-radius: 5px; font-weight: 800; }}
            QTabBar::tab:selected {{ background: #2563EB; }}
            QLabel#AppTitle {{ font-size: {title}px; font-weight: 900; color: #F9FAFB; }}
            QLabel#SensorSelectTitle {{ font-weight: 900; color: #F9FAFB; }}
            QLabel#DisconnectedLabel {{ color: #FCA5A5; font-weight: 900; }}
            QLabel#ConnectedLabel {{ color: #86EFAC; font-weight: 900; }}
            QLabel#QuickStat {{ color: #E5E7EB; font-weight: 900; }}
            QLabel#IdlePill, QLabel#ConnectingPill, QLabel#ReceivingPill, QLabel#ErrorPill, QLabel#StalePill {{
                padding: 4px 10px; border-radius: 8px; font-weight: 900;
            }}
            QLabel#IdlePill {{ background: #374151; color: #E5E7EB; }}
            QLabel#ConnectingPill {{ background: #1D4ED8; color: #DBEAFE; }}
            QLabel#ReceivingPill {{ background: #166534; color: #DCFCE7; }}
            QLabel#ErrorPill {{ background: #991B1B; color: #FEE2E2; }}
            QLabel#StalePill {{ background: #854D0E; color: #FEF3C7; }}
            QLabel#DataAgeLabel, QLabel#SensorValueLabel {{ color: #CBD5E1; font-weight: 900; }}
            QLabel#StatusSummary {{ color: #E5E7EB; font-weight: 800; }}
            QLabel#HintText {{ color: #9CA3AF; font-weight: 700; }}
            QLabel#SectionTitle {{ color: #E5E7EB; font-weight: 900; }}
            QLabel#InferenceSummary {{
                color: #E0F2FE; background: #0C4A6E; border: 1px solid #0369A1;
                border-radius: 7px; padding: 7px; font-weight: 900;
            }}
            QLabel#InfoBadge {{
                color: #CBD5E1; background: #111827; border: 1px solid #374151;
                border-radius: 6px; padding: 4px 6px; font-weight: 800;
            }}
            QLabel#SafetyNote {{
                color: #FDE68A; background: #422006; border: 1px solid #854D0E;
                border-radius: 6px; padding: 5px; font-weight: 700;
            }}
            QLabel#AIIdlePill, QLabel#AIReadyPill, QLabel#AIErrorPill {{
                padding: 5px 9px; border-radius: 7px; font-weight: 900;
            }}
            QLabel#AIIdlePill {{ background: #374151; color: #E5E7EB; }}
            QLabel#AIReadyPill {{ background: #166534; color: #DCFCE7; }}
            QLabel#AIErrorPill {{ background: #991B1B; color: #FEE2E2; }}
            QFrame#StatCard {{ background: #0F172A; border: 1px solid #374151; border-radius: 8px; }}
            QFrame#StatCard[status="ok"] {{ border: 1px solid #15803D; }}
            QFrame#StatCard[status="warning"] {{ border: 1px solid #A16207; }}
            QFrame#StatCard[status="error"] {{ border: 1px solid #B91C1C; }}
            QFrame#ChartPanel {{ background: #0F172A; border: 1px solid #374151; border-radius: 8px; }}
            QLabel#CardTitle {{ color: #9CA3AF; font-size: {max(12, base - 1)}px; font-weight: 800; }}
            QLabel#CardValue {{ color: #F9FAFB; font-size: {card_value}px; font-weight: 900; }}
            QLabel#CardUnit {{ color: #9CA3AF; font-size: {max(11, base - 2)}px; }}
            QLabel#SaveStateLabel {{ font-weight: 900; color: #FDE68A; }}
            QPushButton {{ background: #374151; color: #F9FAFB; border: 1px solid #4B5563; padding: 7px 10px; border-radius: 6px; font-weight: 900; }}
            QPushButton:hover {{ background: #4B5563; }}
            QPushButton:disabled {{ background: #1F2937; color: #6B7280; }}
            QPushButton#PrimaryButton {{ background: #2563EB; border-color: #60A5FA; }}
            QPushButton#PrimaryButton:hover {{ background: #1D4ED8; }}
            QPushButton#SecondaryButton {{ background: #1F2937; color: #CBD5E1; }}
            QPushButton#SensorButton {{ background: #0F172A; color: #E5E7EB; border: 1px solid #4B5563; padding: 5px; border-radius: 6px; font-weight: 900; font-size: {max(15, base + 1)}px; }}
            QPushButton#SensorButton:checked {{ background: #2563EB; color: #FFFFFF; border: 2px solid #93C5FD; }}
            QComboBox, QLineEdit, QPlainTextEdit {{ background: #0F172A; color: #F9FAFB; border: 1px solid #4B5563; padding: 5px; }}
            QGroupBox {{ border: 1px solid #374151; border-radius: 7px; margin-top: 7px; padding: 7px; font-weight: 900; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QProgressBar {{
                background: #0F172A; color: #E5E7EB; border: 1px solid #374151;
                border-radius: 5px; min-height: 16px; text-align: center; font-weight: 800;
            }}
            QProgressBar::chunk {{ background: #2563EB; border-radius: 4px; }}
            """
        )

    def _setup_timer(self) -> None:
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setInterval(250)
        self.plot_timer.timeout.connect(self.refresh_plots)
        self.plot_timer.start()
        self.health_timer = QtCore.QTimer(self)
        self.health_timer.setInterval(500)
        self.health_timer.timeout.connect(self.update_stream_health)
        self.health_timer.start()

    def _setup_ai_worker(self) -> None:
        self.ai_thread = QtCore.QThread(self)
        self.ai_worker = EdgeAIWorker()
        self.ai_worker.moveToThread(self.ai_thread)
        self.ai_thread.finished.connect(self.ai_worker.deleteLater)
        self.ai_row_signal.connect(self.ai_worker.process_row)
        self.ai_load_signal.connect(self.ai_worker.load_package)
        self.ai_unload_signal.connect(self.ai_worker.unload_package)
        self.ai_reset_signal.connect(self.ai_worker.reset_stream)
        self.ai_phase_signal.connect(self.ai_worker.set_experiment_phase)
        self.ai_worker.result_ready.connect(self.on_ai_result)
        self.ai_worker.status_updated.connect(self.on_ai_status)
        self.ai_worker.error_occurred.connect(self.on_ai_error)
        self.ai_worker.package_loaded.connect(self.on_ai_package_loaded)
        self.ai_thread.start()

    # ------------------------- Actions -------------------------
    def choose_ai_package(self) -> None:
        start_dir = str(self.ai_package_path or self.data_root)
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose GAPS deployment package",
            start_dir,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not selected:
            return
        self.ai_package_path = Path(selected).expanduser().resolve()
        self._set_ai_state_visual("AI LOADING", "AIIdlePill")
        self.ai_package_label.setText(self._friendly_model_name(self.ai_package_path.name))
        self.ai_package_label.setToolTip(str(self.ai_package_path))
        self.ai_load_signal.emit(str(self.ai_package_path))

    def refresh_ports(self) -> None:
        current = self.port_combo.currentText() or self.default_port
        ports = []
        try:
            from serial.tools import list_ports
            ports = [p.device for p in list_ports.comports()]
        except Exception:
            ports = []
        fallback = [self.default_port, "/dev/rfcomm0", "/dev/ttyUSB0", "/dev/ttyACM0", "/dev/serial0", "COM3"]
        seen = []
        for item in ports + fallback:
            if item and item not in seen:
                seen.append(item)
        self.port_combo.clear()
        self.port_combo.addItems(seen)
        if current:
            self.port_combo.setCurrentText(current)

    def toggle_connection(self) -> None:
        if self.worker and self.worker.isRunning():
            self.disconnect_serial()
        else:
            self.connect_serial()

    def connect_serial(self) -> None:
        port = self.port_combo.currentText().strip()
        try:
            baud = int(self.baud_combo.currentText())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid baudrate", "Baudrate must be an integer.")
            return
        if not port:
            QtWidgets.QMessageBox.warning(self, "Missing port", "Please select or enter a serial/Bluetooth port.")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.last_error_message = ""
        self.last_frame_wall_time = None
        self._stale_logged = False
        self._disconnect_requested_by_user = False
        self._connection_counter += 1
        self.worker = SerialWorker(
            port=port,
            baudrate=baud,
            connection_id=self._connection_counter,
        )
        self.worker.frame_received.connect(self.on_frame_received)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.connection_changed.connect(self.on_connection_changed)
        self.worker.finished.connect(self.on_serial_worker_finished)
        self._set_stream_state("CONNECTING")
        self._append_event(f"Opening input port: {port} (connection_id={self._connection_counter})", "connection_opening")
        self.worker.start()

    def disconnect_serial(self) -> None:
        worker = self.worker
        if worker is None:
            return
        self._disconnect_requested_by_user = True
        worker.stop()
        if not worker.wait(3000):
            msg = "Serial thread did not stop within 3 seconds. The worker is kept alive to avoid a QThread destruction crash."
            self._append_event("ERROR: " + msg, "serial_stop_timeout")
            self.status_summary_label.setText(msg)
            self._set_stream_state("ERROR")
            return
        self.on_serial_worker_finished()

    @QtCore.pyqtSlot()
    def on_serial_worker_finished(self) -> None:
        sender = self.sender()
        if self.worker is not None and sender not in (None, self.worker):
            return
        self.worker = None
        self.connection_active = False
        self._set_connection_state(False)
        if self.buffer.latest():
            self._set_stream_state("FROZEN")
            self.status_summary_label.setText(
                "Disconnected. The displayed curve is frozen at the last real frame. Existing experiment files are kept."
            )
        else:
            self._set_stream_state("IDLE")
            self.status_summary_label.setText("Disconnected. No data has been received yet.")

    def create_experiment(self, preserve_segment: bool = False) -> None:
        dialog = ExperimentDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            self._append_event("New experiment cancelled. Current experiment is unchanged.")
            return

        if self.segment_rows and not preserve_segment:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved observed segment",
                "The current observed segment has not been cleared and may not have been saved.\n\n"
                "Creating a new experiment will start a clean display/segment boundary. "
                "Save the segment first if it is still needed. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self._append_event("New experiment cancelled to protect the current observed segment.")
                return

        if self.session is not None and self.session.is_open:
            prompt = (
                "A recording is currently running. Stop it and create a NEW experiment?\n\n"
                if self.saving_enabled else
                "A current experiment folder already exists. Close it and create a NEW experiment?\n\n"
            ) + "The current files will be flushed and kept."
            reply = QtWidgets.QMessageBox.question(
                self,
                "Create new experiment",
                prompt,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self._append_event("New experiment cancelled. Current experiment is unchanged.")
                return
            self.saving_enabled = False
            try:
                self.session.log_event("experiment_replaced_by_user")
                self.session.close()
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Close experiment failed", str(exc))
                self._append_event(f"ERROR: Could not close current experiment: {exc}")
                return

        metadata = dialog.metadata()
        metadata["default_input_port"] = self.port_combo.currentText().strip()
        metadata["baudrate"] = self.baud_combo.currentText().strip()
        metadata["edge_ai_package"] = str(self.ai_package_path) if self.ai_package_path else ""
        metadata["edge_ai_package_fingerprint"] = self.last_ai_status.get("package_fingerprint", "")
        metadata["edge_ai_dataset_profile"] = self.last_ai_status.get("dataset_profile", "")
        metadata["edge_ai_device_profile"] = self.last_ai_status.get("device_profile", "")
        metadata["edge_ai_normalization_enabled"] = self.last_ai_status.get(
            "normalization_enabled", ""
        )
        candidate = ExperimentSession(self.data_root, metadata)
        try:
            session_dir = candidate.open()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Create experiment failed", str(exc))
            self._append_event(f"ERROR: Could not create experiment: {exc}")
            return

        self.session = candidate
        self.saving_enabled = False
        self._recording_error_latched = False
        self.save_btn.setText("Start Recording")
        self.save_state_label.setText("Recording: ready")
        self.saved_card.set_value(0, "{}")
        self.exp_dir_label.setText(f"Experiment: {session_dir}")
        if preserve_segment:
            self._append_event("New experiment created while preserving the current observed segment for Save Segment.")
        else:
            self._reset_display_and_segment("New experiment created; display and segment boundary reset.", reset_ai=True)
        self._append_event(f"Experiment created: {session_dir}", "experiment_created")
        self.statusBar().showMessage(f"Experiment created: {session_dir}")
        self._last_disk_check = 0.0
        self.update_disk_space()

    def toggle_saving(self) -> None:
        if self.session is None or not self.session.is_open:
            self.create_experiment()
            if self.session is None or not self.session.is_open:
                return
        try:
            self.saving_enabled = not self.saving_enabled
            if self.saving_enabled:
                self._recording_error_latched = False
                self._last_ai_unsaved_reason = ""
                self.save_btn.setText("Stop Recording")
                self.save_state_label.setText("Recording: ON → raw.csv")
                self.session.log_event("recording_started")
                self._append_event("Continuous recording started. New real frames will be written to raw.csv.")
            else:
                self.save_btn.setText("Start Recording")
                self.save_state_label.setText("Recording: stopped")
                self.session.log_event("recording_stopped")
                self._append_event("Continuous recording stopped.")
        except Exception as exc:
            self.saving_enabled = False
            self.save_btn.setText("Start Recording")
            self.save_state_label.setText("Recording: ERROR")
            self._append_event(f"ERROR: Failed to change recording state: {exc}")
            QtWidgets.QMessageBox.critical(self, "Recording error", str(exc))

    def save_current_segment(self) -> None:
        if not self.segment_rows:
            QtWidgets.QMessageBox.information(self, "No data", "No received frames are available to save yet.")
            return
        if self.session is None or not self.session.is_open:
            self.create_experiment(preserve_segment=True)
            if self.session is None or not self.session.is_open:
                return
        if not self.segment_rows:
            QtWidgets.QMessageBox.warning(self, "No segment", "The observed segment is empty; nothing was saved.")
            return
        assert self.session.session_dir is not None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.session.session_dir / f"saved_segment_{stamp}.csv"
        meta_path = self.session.session_dir / f"saved_segment_{stamp}.json"
        columns = list(CSV_COLUMNS_WITH_DERIVED)
        rows_snapshot = list(self.segment_rows)
        try:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                for row in rows_snapshot:
                    writer.writerow({col: row.get(col, "") for col in columns})
            segment_meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "csv_file": path.name,
                "rows": len(rows_snapshot),
                "selected_sensor": self.selected_sensor_index + 1,
                "start_timestamp_iso": rows_snapshot[0].get("timestamp_iso", ""),
                "end_timestamp_iso": rows_snapshot[-1].get("timestamp_iso", ""),
                "start_elapsed_s": rows_snapshot[0].get("elapsed_s", ""),
                "end_elapsed_s": rows_snapshot[-1].get("elapsed_s", ""),
                "truncated": bool(self._segment_truncated_rows),
                "dropped_oldest_rows": int(self._segment_truncated_rows),
                "note": (
                    "Segment rows are frames collected since the last Mark Segment Start/Clear operation."
                    if not self._segment_truncated_rows else
                    "The in-memory segment cap was reached; this file contains the newest retained rows only. raw.csv remains complete when recording was enabled."
                ),
            }
            meta_path.write_text(json.dumps(segment_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            msg = f"Failed to save segment: {exc}"
            self._append_event("ERROR: " + msg)
            QtWidgets.QMessageBox.critical(self, "Save segment failed", msg)
            return

        n = len(rows_snapshot)
        try:
            self.session.log_event("segment_saved", f"{path.name}, rows={n}")
        except Exception as exc:
            self._append_event(f"WARNING: Segment saved but event log update failed: {exc}")
        self._append_event(f"Observed segment saved: {path.name} ({n} rows) with metadata {meta_path.name}.")
        self.statusBar().showMessage(f"Saved observed segment: {path}")

        reply = QtWidgets.QMessageBox.question(
            self,
            "Segment saved",
            "Segment saved successfully. Start a new observed segment now?\n\n"
            "Yes will clear the current display/segment buffer. Saved files are not deleted.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._reset_display_and_segment("New observed segment started after saving previous segment.")

    def clear_curves(self) -> None:
        if len(self.segment_rows) > 0 or bool(self.buffer.latest()):
            reply = QtWidgets.QMessageBox.question(
                self,
                "Mark new segment start",
                "Reset the displayed curve and start a NEW observed segment from the next received frame?\n\n"
                "This will NOT delete raw.csv, saved_segment files, meta.json, or event_log.csv.\n"
                "If you still need the current observed segment, click No and use Save Segment first.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self._append_event("Segment reset cancelled by user.")
                return
        self._reset_display_and_segment("New observed segment marked by user. Saved files were not deleted.")

    def _reset_display_and_segment(self, message: str, reset_ai: bool = False) -> None:
        self.buffer.clear()
        self.segment_rows.clear()
        self._segment_truncated_rows = 0
        self.sensor_curve.setData([], [])
        self.sample_label.setText("Latest: --")
        self.sensor_value_label.setText(f"Sensor {self.selected_sensor_index + 1}  Latest: --")
        self.segment_card.set_value(0, "{}")
        if reset_ai:
            self.ai_reset_signal.emit(False)
        self._append_event(message, "segment_boundary")

    def open_data_folder(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.data_root.resolve())))

    def choose_save_path(self) -> None:
        """Choose the default root directory for future experiment folders."""
        if self.session is not None and self.session.is_open:
            msg = QtWidgets.QMessageBox.question(
                self,
                "Change save path",
                "A current experiment already exists. Changing the save path will only affect NEW experiments.\n\n"
                "Current experiment files will stay in their existing folder. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if msg != QtWidgets.QMessageBox.Yes:
                return
        self.data_root.mkdir(parents=True, exist_ok=True)
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Choose default save folder",
            str(self.data_root),
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if not selected:
            return
        self.data_root = Path(selected).expanduser().resolve()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._append_event(f"Default save path set to: {self.data_root}")
        self._last_disk_check = 0.0
        self.update_disk_space()
        if self.session is None or not self.session.is_open:
            self.exp_dir_label.setText(f"Experiment root: {self.data_root}")
        self.statusBar().showMessage(f"Default save path: {self.data_root}")

    def mark_experiment_event(self, event_name: str, label: str) -> None:
        phase_map = {
            "baseline_start": "baseline",
            "exposure_start": "exposure",
            "recovery_start": "recovery",
        }
        ai_phase = phase_map.get(event_name)
        if ai_phase is not None:
            self.ai_phase_signal.emit(ai_phase)
        latest = self.buffer.latest()
        context = {
            "stream_frame_index": self._stream_frame_index - 1 if self._stream_frame_index else -1,
            "stream_elapsed_s": latest.get("elapsed_s", "") if latest else "",
            "selected_sensor": self.selected_sensor_index + 1,
            "ai_gas": self.last_ai_result.get("predicted_gas", ""),
            "ai_ppm": self.last_ai_result.get("ppm_full_prediction", ""),
            "ai_qc": self.last_ai_result.get("decision", ""),
            "ai_phase": ai_phase or self.last_ai_status.get("experiment_phase", ""),
        }
        self._append_event(f"{label} | {json.dumps(context, ensure_ascii=False)}", event_name)
        self.statusBar().showMessage(label)

    def add_experiment_note(self) -> None:
        text, ok = QtWidgets.QInputDialog.getMultiLineText(self, "Experiment note", "Note")
        if ok and text.strip():
            self._append_event(text.strip(), "operator_note")

    def select_sensor(self, idx: int) -> None:
        idx = int(max(0, min(15, idx)))
        self.selected_sensor_index = idx
        for i, btn in enumerate(self.sensor_buttons):
            btn.setChecked(i == idx)
        self.sensor_plot.setTitle(f"Sensor {idx + 1}")
        self.sensor_curve.setPen(pg.mkPen(pg.intColor(idx, hues=16), width=2.8))
        if hasattr(self, "selected_card"):
            self.selected_card.set_value(idx + 1, "{}")
        latest = self.buffer.latest()
        if latest:
            name = ADC_FIELD_NAMES[idx]
            val = latest.get(name, 0.0)
            self.sensor_value_label.setText(f"Sensor {idx + 1}  Latest: {val:.0f} raw")
        else:
            self.sensor_value_label.setText(f"Sensor {idx + 1}  Latest: --")
        self.refresh_plots()

    # ------------------------- Slots -------------------------
    @QtCore.pyqtSlot(dict)
    def on_frame_received(self, row: Dict[str, object]) -> None:
        self.last_frame_wall_time = time.time()
        self._stale_logged = False
        ts = float(row.get("timestamp_unix", time.time()))
        if self._stream_start_timestamp is None:
            self._stream_start_timestamp = ts
        stream_elapsed = max(0.0, ts - self._stream_start_timestamp)
        row = dict(row)
        row["stream_frame_index"] = self._stream_frame_index
        row["stream_elapsed_s"] = f"{stream_elapsed:.6f}"
        # Preserve backward compatibility while making the legacy fields reconnect-safe.
        row["frame_index"] = self._stream_frame_index
        row["elapsed_s"] = f"{stream_elapsed:.6f}"
        self._stream_frame_index += 1

        self.buffer.append_row(row)
        if self.max_segment_rows > 0 and len(self.segment_rows) >= self.max_segment_rows:
            overflow = len(self.segment_rows) - self.max_segment_rows + 1
            if overflow > 0:
                del self.segment_rows[:overflow]
                self._segment_truncated_rows += overflow
        self.segment_rows.append(dict(row))

        latest = self.buffer.latest()
        if latest:
            self.rh_card.set_value(latest.get("aht20_rh", 0.0), "{:.0f}")
            self.temp_card.set_value(latest.get("aht20_temp_c", 0.0), "{:.0f}")
            aux4 = int(float(latest.get("uart4_wz_h3_nk_ppb", 0.0)))
            aux5 = int(float(latest.get("uart5_tb200b_ppb", 0.0)))
            self.aux_label.setText(f"UART4/UART5 raw: {aux4}/{aux5}")
            selected_name = ADC_FIELD_NAMES[self.selected_sensor_index]
            val = latest.get(selected_name, 0.0)
            self.sensor_value_label.setText(f"Sensor {self.selected_sensor_index + 1}  Latest: {val:.0f} raw")
            self.sample_label.setText(f"Latest: t={latest.get('elapsed_s', 0):.1f}s")
            segment_text = str(len(self.segment_rows))
            if self._segment_truncated_rows:
                segment_text += "*"
            self.segment_card.set_value(segment_text, "{}")

        frame_plausible = int(float(row.get("frame_plausible", 1))) != 0
        if not frame_plausible:
            now = time.time()
            if now - self._last_implausible_event_time >= 5.0:
                self._last_implausible_event_time = now
                self._append_event(
                    f"WARNING: structurally valid frame has suspicious values: {row.get('plausibility_issue', '')}",
                    "frame_implausible",
                )

        recording_active = bool(
            self.saving_enabled
            and self.session is not None
            and self.session.is_open
            and self.session.session_dir is not None
        )
        row["_recording_active"] = recording_active
        row["_recording_session_id"] = (
            str(self.session.session_dir) if recording_active and self.session is not None else ""
        )

        if self.saving_enabled and self.session and self.session.is_open:
            try:
                self.session.write_row(row)
                self.saved_card.set_value(self.session.rows_written, "{}")
            except Exception as exc:
                self.saving_enabled = False
                self._recording_error_latched = True
                self.save_btn.setText("Start Recording")
                self.save_state_label.setText("Recording: ERROR - stopped")
                msg = f"CSV write failed. Recording stopped to protect data. Reason: {exc}"
                self._append_event("ERROR: " + msg, "recording_write_error")
                self.status_summary_label.setText(msg)
                self._set_stream_state("ERROR")
                return

        # The package runtime counts implausible frames and fails closed by default.
        self.ai_row_signal.emit(dict(row))
        if not self._recording_error_latched:
            self._set_stream_state("RECEIVING")
            self.status_summary_label.setText(
                "Receiving real frames. Plot, recording and optional edge AI use the same parsed stream."
            )

    @QtCore.pyqtSlot(dict)
    def on_stats_updated(self, stats: Dict[str, object]) -> None:
        self.last_stats = stats
        good = int(stats.get("good_frames", 0))
        bad = int(stats.get("bad_frames", 0))
        fps = float(stats.get("fps", 0.0))
        self.good_label.setText(str(good))
        self.bad_label.setText(str(bad))
        self.fps_label.setText(f"{fps:.3f}")
        self.drop_label.setText(str(stats.get("dropped_bytes", 0)))
        self.resync_label.setText(str(stats.get("resync_count", 0)))
        self.buffered_label.setText(str(stats.get("buffered_bytes", 0)))
        self.implausible_label.setText(str(stats.get("implausible_frames", 0)))
        self.quick_good_label.setText(f"G:{good}")
        self.quick_bad_label.setText(f"B:{bad}")
        self.quick_fps_label.setText(f"{fps:.2f} fps")

    @QtCore.pyqtSlot(str)
    def on_error(self, message: str) -> None:
        friendly = self._friendly_error(message)
        self.last_error_message = friendly
        self.statusBar().showMessage(friendly)
        self._append_event("ERROR: " + friendly, "serial_error")
        self._set_stream_state("ERROR")
        self.status_summary_label.setText(friendly)
        self._set_connection_state(False)

    @QtCore.pyqtSlot(bool, str)
    def on_connection_changed(self, connected: bool, port: str) -> None:
        self.connection_active = bool(connected)
        self._set_connection_state(connected)
        event_name = "serial_connected" if connected else "serial_disconnected"
        msg = ("Connected to " if connected else "Disconnected from ") + port
        self.statusBar().showMessage(msg)
        self._append_event(msg, event_name)
        if connected:
            self._disconnect_requested_by_user = False
            self._set_stream_state("CONNECTING")
            self.status_summary_label.setText("Connected. Waiting for valid 43-byte STM32 frames...")
        else:
            # A model window must never bridge a physical input disconnect.
            self.ai_reset_signal.emit(True)
            if self.saving_enabled:
                self.saving_enabled = False
                self.save_btn.setText("Start Recording")
                self.save_state_label.setText("Recording: paused after disconnect")
                self._append_event(
                    "Recording was paused because the input disconnected. Reconnect and start recording explicitly to continue.",
                    "recording_paused_disconnect",
                )
            if not self.last_error_message:
                if self.buffer.latest():
                    self._set_stream_state("FROZEN")
                    self.status_summary_label.setText("Input disconnected. Display is frozen at the last received curve.")
                else:
                    self._set_stream_state("IDLE")
            self.connection_active = False

    @QtCore.pyqtSlot(bool, str)
    def on_ai_package_loaded(self, loaded: bool, message: str) -> None:
        if loaded:
            self._set_ai_state_visual("AI READY", "AIReadyPill")
            self.ai_package_label.setText(self._friendly_model_name(message))
            self.ai_package_label.setToolTip(str(message))
            self._ai_history_ppm.clear()
            self._ai_history_index.clear()
            self.ai_history_curve.setData([], [])
            self._append_event(f"Edge AI package loaded: {message}", "ai_package_loaded")
        else:
            self.last_ai_status = {}
            self._last_ai_fingerprint_logged = ""
            self._set_ai_state_visual("AI OFFLINE", "AIIdlePill")
            self.ai_package_label.setText("No active model")
            self.ai_package_label.setToolTip(str(message))
            self.ai_result_label.setText("Waiting for a verified model package.")
            self._append_event(f"Edge AI package not active: {message}", "ai_package_unloaded")

    @QtCore.pyqtSlot(dict)
    def on_ai_status(self, status: Dict[str, object]) -> None:
        self.last_ai_status = dict(status)
        state = str(status.get("state", "unknown"))
        warm_n = int(status.get("warmup_collected", 0))
        warm_req = int(status.get("warmup_required", 0))
        base_n = int(status.get("baseline_collected", 0))
        base_req = int(status.get("baseline_required", 0))
        win_n = int(status.get("window_collected", 0))
        win_req = int(status.get("window_required", 0))
        observed = float(status.get("observed_hz", 0.0))
        expected = float(status.get("expected_hz", 0.0))
        mode = str(status.get("feature_mode", ""))
        phase = str(status.get("experiment_phase", ""))
        dataset_profile = str(status.get("dataset_profile", "unspecified"))
        device_profile = str(status.get("device_profile", "unspecified"))
        normalized = bool(status.get("normalization_enabled", False))
        fingerprint = str(status.get("package_fingerprint", ""))
        self.ai_rate_card.set_value(observed, "{:.2f}")
        phase_text = {
            "automatic": "Auto",
            "baseline": "Baseline",
            "exposure": "Exposure",
            "recovery": "Recovery",
        }.get(phase.lower(), phase.title() or "--")
        self.ai_phase_badge.setText(f"Phase: {phase_text}")
        if base_req <= 0 or base_n >= base_req:
            baseline_text = "Ready"
        else:
            baseline_text = f"{base_n}/{base_req}"
        self.ai_baseline_badge.setText(f"Baseline: {baseline_text}")
        self.ai_window_badge.setText(f"Window: {win_n}/{win_req}")
        rate_text = (
            f"{observed:.2f}/{expected:.2f} Hz"
            if expected > 0
            else f"{observed:.2f} Hz"
        )
        self.ai_rate_badge.setText(f"Rate: {rate_text}")
        self.ai_window_progress.setRange(0, max(1, win_req))
        self.ai_window_progress.setValue(min(max(0, win_n), max(1, win_req)))
        self.ai_window_progress.setFormat(
            f"Inference window: %v/{win_req if win_req > 0 else '--'}"
        )
        technical_details = (
            f"state={state}; phase={phase}; baseline={base_n}/{base_req}; "
            f"window={win_n}/{win_req}; input={observed:.4f}/{expected:.4f} Hz; "
            f"dataset={dataset_profile}; device={device_profile}; mode={mode}; "
            f"normalization={'on' if normalized else 'off'}"
        )
        self.ai_progress_label.setToolTip(technical_details)
        package_name = str(status.get("package_name", "") or "")
        if package_name:
            self.ai_package_label.setText(self._friendly_model_name(package_name))
            self.ai_package_label.setToolTip(
                f"{package_name}\n{technical_details}\nsha256={fingerprint}"
            )
        if fingerprint and fingerprint != self._last_ai_fingerprint_logged:
            self._last_ai_fingerprint_logged = fingerprint
            self._append_event(
                "AI package audit: "
                f"dataset={dataset_profile}, device={device_profile}, "
                f"normalization={normalized}, sha256={fingerprint}",
                "ai_package_audit",
            )
            if self.session is not None and self.session.is_open:
                try:
                    self.session.update_metadata(
                        {
                            "edge_ai_runtime": {
                                "package_name": status.get("package_name", ""),
                                "package_fingerprint": fingerprint,
                                "dataset_profile": dataset_profile,
                                "device_profile": device_profile,
                                "model_backend": status.get("model_backend", ""),
                                "runtime_v5_release_id": status.get(
                                    "runtime_v5_release_id", ""
                                ),
                                "runtime_v5_qc_status": status.get(
                                    "runtime_v5_qc_status", ""
                                ),
                                "normalization_enabled": normalized,
                                "expected_hz": expected,
                                "feature_mode": mode,
                                "schema_version": status.get("schema_version", ""),
                            }
                        }
                    )
                except Exception as exc:
                    self._append_event(
                        f"WARNING: Could not update AI provenance in meta.json: {exc}",
                        "ai_metadata_update_error",
                    )
        if state == "rate_mismatch":
            self.ai_progress_label.setText(
                f"Input-rate mismatch: {observed:.2f} Hz received, "
                f"{expected:.2f} Hz required. Inference is blocked."
            )
            self.ai_rate_card.set_status("error")
        elif state in {"waiting_for_baseline_phase", "baseline_required"}:
            self.ai_progress_label.setText(
                "Baseline required: mark Baseline Start and collect clean-air samples."
            )
            self.ai_rate_card.set_status("ok")
        elif warm_req > 0 and warm_n < warm_req:
            self.ai_progress_label.setText(
                f"Sensor stabilization: {warm_n}/{warm_req} samples."
            )
            self.ai_rate_card.set_status("ok")
        elif base_req > 0 and base_n < base_req:
            self.ai_progress_label.setText(
                f"Collecting baseline: {base_n}/{base_req} samples."
            )
            self.ai_rate_card.set_status("ok")
        else:
            self.ai_progress_label.setText(
                "Runtime ready. Predictions use the verified model package shown above."
            )
            self.ai_rate_card.set_status("ok")
        self._update_ai_detail_visibility()

    @QtCore.pyqtSlot(dict)
    def on_ai_result(self, result: Dict[str, object]) -> None:
        self.last_ai_result = dict(result)
        gas = str(result.get("predicted_gas", "--"))
        ppm = float(result.get("ppm_full_prediction", 0.0))
        conf = float(result.get("confidence", 0.0))
        decision = str(result.get("decision", "review"))
        latency = float(result.get("inference_latency_ms", 0.0))
        risk = result.get("risk_score")
        risk_text = "--" if risk in (None, "") else f"{float(risk):.3f}"
        self.ai_gas_card.set_value(gas, "{}")
        self.ai_ppm_card.set_value(ppm, "{:.1f}")
        self.ai_conf_card.set_value(conf * 100.0, "{:.1f}")
        qc_display, qc_status = self._friendly_qc_decision(decision)
        self.ai_qc_card.set_value(qc_display, "{}")
        self.ai_qc_card.value.setToolTip(f"Runtime decision: {decision}")
        self.ai_qc_card.set_status(qc_status)
        self.ai_latency_card.set_value(latency, "{:.1f}")
        self.ai_gas_card.set_status("ok")
        self.ai_ppm_card.set_status("ok")
        self.ai_conf_card.set_status("ok")
        self.ai_latency_card.set_status("ok")
        auto = result.get("ppm_auto_output")
        if decision == "disabled_pending_dependency_audit":
            auto_text = "QC disabled; no auto output"
        else:
            auto_text = (
                "manual review"
                if auto in (None, "")
                else f"auto={float(auto):.1f} ppm"
            )
        self.ai_result_label.setText(
            f"{gas} · full prediction {ppm:.1f} ppm · {auto_text}"
        )
        self.ai_result_label.setToolTip(
            f"confidence={conf:.6f}; risk={risk_text}; QC={decision}"
        )
        next_index = self._ai_history_index[-1] + 1 if self._ai_history_index else 1
        self._ai_history_index.append(next_index)
        self._ai_history_ppm.append(ppm)
        self._ai_history_index = self._ai_history_index[-120:]
        self._ai_history_ppm = self._ai_history_ppm[-120:]
        self.ai_history_curve.setData(
            self._ai_history_index,
            self._ai_history_ppm,
        )
        active_session_id = (
            str(self.session.session_dir)
            if self.session is not None
            and self.session.is_open
            and self.session.session_dir is not None
            else ""
        )
        result_session_id = str(result.get("recording_session_id", ""))
        recording_complete = bool(result.get("window_recording_complete", False))
        if (
            recording_complete
            and active_session_id
            and result_session_id == active_session_id
            and self.session is not None
        ):
            try:
                self.session.write_ai_result(result)
                self._last_ai_unsaved_reason = ""
            except Exception as exc:
                self._append_event(f"ERROR: Edge AI output could not be saved: {exc}", "ai_output_write_error")
        elif self.saving_enabled:
            reason = (
                "AI window was not fully covered by one active raw.csv recording session; "
                "prediction remains visible but is not written to the experiment audit CSV."
            )
            if reason != self._last_ai_unsaved_reason:
                self._last_ai_unsaved_reason = reason
                self._append_event(reason, "ai_output_not_auditable")

    @QtCore.pyqtSlot(str)
    def on_ai_error(self, message: str) -> None:
        self._set_ai_state_visual("AI ERROR", "AIErrorPill")
        self.ai_result_label.setText(message)
        self._append_event("ERROR: Edge AI: " + message, "ai_error")

    def _set_connection_state(self, connected: bool) -> None:
        compact = bool(self._compact_ui)
        self.connect_btn.setText(
            "Disconnect" if connected else ("Connect" if compact else "Connect HC-04")
        )
        self.port_combo.setEnabled(not connected)
        self.baud_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)
        self.connection_label.setText("Connected" if connected else "Disconnected")
        self.connection_label.setObjectName("ConnectedLabel" if connected else "DisconnectedLabel")
        self.connection_label.style().unpolish(self.connection_label)
        self.connection_label.style().polish(self.connection_label)

    def _append_event(self, message: str, event: str = "ui_event") -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            self.event_log_view.appendPlainText(f"[{stamp}] {message}")
        except Exception:
            pass
        if self.session is not None and self.session.is_open:
            try:
                self.session.log_event(event, message)
            except Exception:
                # Avoid recursive logging if the event log itself becomes unavailable.
                pass

    def _set_stream_state(self, state: str) -> None:
        state = state.upper()
        mapping = {
            "IDLE": ("IDLE", "IdlePill"),
            "CONNECTING": ("WAIT", "ConnectingPill"),
            "RECEIVING": ("LIVE", "ReceivingPill"),
            "ERROR": ("ERROR", "ErrorPill"),
            "STALE": ("NO FRAME", "StalePill"),
            "FROZEN": ("FROZEN", "StalePill"),
        }
        text, obj = mapping.get(state, (state, "IdlePill"))
        self.stream_state_label.setText(text)
        self.stream_state_label.setObjectName(obj)
        self.stream_state_label.style().unpolish(self.stream_state_label)
        self.stream_state_label.style().polish(self.stream_state_label)

    def _friendly_error(self, message: str) -> str:
        low = str(message).lower()
        if "host is down" in low:
            return "HC-04 is not responding. It may be occupied by phone/Windows. Disconnect other hosts or power-cycle the module."
        if "permission" in low or "denied" in low:
            return "Serial permission denied. Add the user to dialout or check the device permission."
        if "no such file" in low or "could not open port" in low:
            return "Input port not found. Check Bluetooth/USB connection and the Port field."
        if "resource busy" in low or "busy" in low:
            return "Input port is busy. Another program may be using HC-04 or the serial port. Close it and reconnect."
        return str(message)

    def update_stream_health(self) -> None:
        self.update_disk_space()
        if self.last_frame_wall_time is None:
            self.data_age_label.setText("No data")
            if self.connection_active:
                self._set_stream_state("CONNECTING")
            return
        age = max(0.0, time.time() - self.last_frame_wall_time)
        self.data_age_label.setText(f"{age:.1f}s ago")
        expected_hz = float(self.last_ai_status.get("expected_hz", 0.0) or 0.0)
        stale_after_s = max(3.0, 5.0 / expected_hz) if expected_hz > 0 else 3.0
        if self.connection_active and age > stale_after_s:
            self._set_stream_state("STALE")
            self.status_summary_label.setText(
                f"Connected, but no valid frame arrived for more than {stale_after_s:.1f} seconds. "
                "Check HC-04, /dev/rfcomm0 and STM32 output."
            )
            if not self._stale_logged:
                self._stale_logged = True
                self._append_event(f"No valid frame for {age:.1f}s", "stream_stale")
        elif self.connection_active and age <= stale_after_s:
            self._stale_logged = False

    def update_disk_space(self) -> None:
        now = time.time()
        if now - self._last_disk_check < 5.0:
            return
        self._last_disk_check = now
        try:
            check_path = (
                self.session.session_dir
                if self.session is not None and self.session.is_open and self.session.session_dir is not None
                else self.data_root
            )
            Path(check_path).mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(str(check_path))
            free_gb = usage.free / (1024 ** 3)
            self.disk_card.set_value(free_gb, "{:.1f}")
            if free_gb < 0.20 and self.saving_enabled:
                self.saving_enabled = False
                self.save_btn.setText("Start Recording")
                self.save_state_label.setText("Recording: stopped — critically low disk")
                self._append_event(
                    f"CRITICAL: recording stopped because only {free_gb:.2f} GB remains on {check_path}",
                    "recording_stopped_low_disk",
                )
            elif free_gb < 1.0 and not self._low_disk_warned:
                self._low_disk_warned = True
                self._append_event(
                    f"WARNING: Low disk space: {free_gb:.2f} GB free on {check_path}",
                    "low_disk_warning",
                )
            elif free_gb >= 1.5:
                self._low_disk_warned = False
        except Exception as exc:
            self.disk_card.set_value("ERR", "{}")
            if not self._low_disk_warned:
                self._low_disk_warned = True
                self._append_event(f"WARNING: Could not check disk space: {exc}", "disk_check_error")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self.isFullScreen():
            self.showMaximized()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_F11:
            self.showMaximized() if self.isFullScreen() else self.showFullScreen()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Q and (event.modifiers() & QtCore.Qt.ControlModifier):
            self.close()
            event.accept()
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def refresh_plots(self) -> None:
        xs, ys = self.buffer.plot_arrays()
        name = ADC_FIELD_NAMES[self.selected_sensor_index]
        self.sensor_curve.setData(xs, ys.get(name, []))

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._closing:
            event.accept()
            return
        if self.saving_enabled:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Close while recording",
                "Recording is still ON. Stop recording and close the program?\n\n"
                "raw.csv and edge_ai_predictions.csv will be flushed safely.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        self._closing = True
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(3000):
                self._closing = False
                QtWidgets.QMessageBox.critical(
                    self,
                    "Serial thread still running",
                    "The serial thread did not stop safely. The application will remain open to avoid a QThread crash.",
                )
                event.ignore()
                return
        if self.session and self.session.is_open:
            self.saving_enabled = False
            try:
                self.session.close()
            except Exception as exc:
                self._append_event(f"ERROR while closing experiment: {exc}")
        if self.ai_thread is not None:
            self.ai_thread.quit()
            if not self.ai_thread.wait(3000):
                self._closing = False
                QtWidgets.QMessageBox.critical(
                    self,
                    "AI thread still running",
                    "The AI thread did not stop safely. The application will remain open; "
                    "wait for inference to finish and close again.",
                )
                event.ignore()
                return
        event.accept()


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults: Dict[str, object] = {}
    if known.config:
        defaults = ui_defaults(load_config(known.config))

    parser = argparse.ArgumentParser(
        description="Raspberry Pi STM32 upper-computer v2.0 with guarded recording and optional GAPS edge-AI inference."
    )
    parser.add_argument("--config", default=known.config, help="Optional YAML/JSON config file.")
    parser.add_argument(
        "--data-root",
        default=str(defaults.get("data_root", DEFAULT_DATA_ROOT)),
        help="Root directory for experiment folders.",
    )
    parser.add_argument(
        "--port",
        default=str(defaults.get("port", DEFAULT_HC04_PORT)),
        help="Serial input device. HC-04 default: /dev/rfcomm0.",
    )
    parser.add_argument("--baudrate", type=int, default=int(defaults.get("baudrate", 115200)))
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=int(defaults.get("max_plot_points", 1800)),
        help="Recent points retained for plotting. raw.csv remains disk-streamed.",
    )
    parser.add_argument(
        "--max-segment-rows",
        type=int,
        default=int(defaults.get("max_segment_rows", 100000)),
        help="In-memory segment cap. Oldest rows are dropped with explicit metadata; raw.csv is unaffected.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=bool(defaults.get("fullscreen", False)),
        help="Show UI fullscreen on the Raspberry Pi screen.",
    )
    parser.add_argument("--font-scale", type=float, default=float(defaults.get("font_scale", 1.12)))
    parser.add_argument(
        "--ai-package",
        default=str(defaults.get("ai_package", "")),
        help=(
            "Optional GAPS deployment package directory. Supports schema-v1/v2 "
            "TorchScript and schema-v3 Runtime-v5 packages."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("GAPS Edge AI Monitor")
    win = MainWindow(
        data_root=Path(args.data_root),
        default_port=args.port,
        default_baudrate=args.baudrate,
        max_plot_points=args.max_plot_points,
        max_segment_rows=args.max_segment_rows,
        fullscreen=args.fullscreen,
        font_scale=args.font_scale,
        ai_package=Path(args.ai_package) if args.ai_package else None,
    )
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.showMaximized()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
