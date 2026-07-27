"""Qt worker that keeps optional edge-AI preprocessing and inference off the UI thread."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5 import QtCore

from edge_ai_runtime import EdgeAIRuntime


class EdgeAIWorker(QtCore.QObject):
    result_ready = QtCore.pyqtSignal(dict)
    status_updated = QtCore.pyqtSignal(dict)
    error_occurred = QtCore.pyqtSignal(str)
    package_loaded = QtCore.pyqtSignal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.runtime: Optional[EdgeAIRuntime] = None
        self.enabled = False
        self._consecutive_errors = 0

    @QtCore.pyqtSlot(str)
    def load_package(self, package_dir: str) -> None:
        try:
            self.runtime = EdgeAIRuntime(Path(package_dir))
            self.enabled = True
            self._consecutive_errors = 0
            self.package_loaded.emit(True, self.runtime.package.package_name)
            self.status_updated.emit(self.runtime.status())
        except Exception as exc:
            self.runtime = None
            self.enabled = False
            self.package_loaded.emit(False, str(exc))
            self.error_occurred.emit(str(exc))

    @QtCore.pyqtSlot()
    def unload_package(self) -> None:
        self.runtime = None
        self.enabled = False
        self._consecutive_errors = 0
        self.package_loaded.emit(False, "AI package unloaded")

    @QtCore.pyqtSlot(bool)
    def reset_stream(self, keep_baseline: bool = False) -> None:
        if self.runtime is None:
            return
        self.runtime.reset_stream(keep_baseline=bool(keep_baseline))
        self.status_updated.emit(self.runtime.status())

    @QtCore.pyqtSlot(str)
    def set_experiment_phase(self, phase: str) -> None:
        if self.runtime is None:
            return
        try:
            self.runtime.set_experiment_phase(phase)
            self.status_updated.emit(self.runtime.status())
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    @QtCore.pyqtSlot(dict)
    def process_row(self, row: dict) -> None:
        if not self.enabled or self.runtime is None:
            return
        try:
            result = self.runtime.append_row(row)
            self.status_updated.emit(self.runtime.status())
            if result is not None:
                self._consecutive_errors = 0
                self.result_ready.emit(result.to_dict())
        except Exception as exc:
            self._consecutive_errors += 1
            self.error_occurred.emit(str(exc))
            if self._consecutive_errors >= 3:
                self.enabled = False
                self.package_loaded.emit(
                    False,
                    "AI disabled after 3 consecutive runtime errors; reload the package after fixing it",
                )
