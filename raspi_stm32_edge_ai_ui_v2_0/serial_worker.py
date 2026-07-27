"""Qt serial worker for reading STM32 frames without blocking the UI."""

from __future__ import annotations

import threading
import time
from typing import Optional

from PyQt5 import QtCore

from frame_parser_v20 import STM32FrameParserV20


class SerialWorker(QtCore.QThread):
    frame_received = QtCore.pyqtSignal(dict)
    stats_updated = QtCore.pyqtSignal(dict)
    error_occurred = QtCore.pyqtSignal(str)
    connection_changed = QtCore.pyqtSignal(bool, str)

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        chunk_size: int = 256,
        stats_interval_s: float = 1.0,
        connection_id: int = 0,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.chunk_size = int(chunk_size)
        self.stats_interval_s = float(stats_interval_s)
        self.connection_id = int(connection_id)
        self._stop_requested = threading.Event()
        self._serial = None
        self._serial_lock = threading.Lock()
        self.parser = STM32FrameParserV20()
        self._first_frame_ts: Optional[float] = None
        self._start_time: Optional[float] = None

    def stop(self) -> None:
        """Request exit and actively interrupt a blocking serial read when possible."""
        self._stop_requested.set()
        with self._serial_lock:
            ser = self._serial
        if ser is None:
            return
        try:
            cancel_read = getattr(ser, "cancel_read", None)
            if callable(cancel_read):
                cancel_read()
        except Exception:
            pass
        try:
            ser.close()
        except Exception:
            pass

    def _stats_dict(self) -> dict:
        now = time.time()
        elapsed = max(now - (self._start_time or now), 1e-9)
        return {
            "good_frames": self.parser.stats.good_frames,
            "bad_frames": self.parser.stats.bad_frames,
            "implausible_frames": self.parser.stats.implausible_frames,
            "dropped_bytes": self.parser.stats.dropped_bytes,
            "resync_count": self.parser.stats.resync_count,
            "buffered_bytes": self.parser.buffered_bytes,
            "fps": self.parser.stats.good_frames / elapsed,
            "port": self.port,
            "baudrate": self.baudrate,
            "connection_id": self.connection_id,
        }

    def run(self) -> None:
        ser = None
        try:
            import serial
            ser = serial.serial_for_url(
                url=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                do_not_open=True,
            )
            with self._serial_lock:
                self._serial = ser
            ser.open()
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.error_occurred.emit(f"Failed to open serial port {self.port}: {exc}")
            self.connection_changed.emit(False, self.port)
            with self._serial_lock:
                self._serial = None
            return

        self._stop_requested.clear()
        self.parser.reset()
        self._first_frame_ts = None
        self._start_time = time.time()
        last_stats = self._start_time
        self.connection_changed.emit(True, self.port)

        try:
            while not self._stop_requested.is_set():
                try:
                    chunk = ser.read(self.chunk_size)
                except Exception:
                    if self._stop_requested.is_set():
                        break
                    raise
                now = time.time()
                if chunk:
                    frames = self.parser.feed(chunk, timestamp_unix=now)
                    for frame in frames:
                        if self._first_frame_ts is None:
                            self._first_frame_ts = frame.timestamp_unix
                        row = frame.as_csv_dict(start_timestamp=self._first_frame_ts, derived=True)
                        row["connection_id"] = self.connection_id
                        row["connection_frame_index"] = frame.frame_index
                        row["connection_elapsed_s"] = row.get("elapsed_s", "0")
                        self.frame_received.emit(row)
                if now - last_stats >= self.stats_interval_s:
                    self.stats_updated.emit(self._stats_dict())
                    last_stats = now
        except Exception as exc:  # pragma: no cover - runtime guard
            self.error_occurred.emit(str(exc))
        finally:
            try:
                if ser is not None:
                    ser.close()
            except Exception:
                pass
            with self._serial_lock:
                self._serial = None
            self.stats_updated.emit(self._stats_dict())
            self.connection_changed.emit(False, self.port)
