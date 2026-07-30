"""Experiment-session writers for raw STM32 data, events and edge-AI outputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Dict, Optional

from frame_parser_v20 import CSV_COLUMNS_WITH_DERIVED


AI_OUTPUT_COLUMNS = [
    "timestamp_iso", "stream_frame_index", "predicted_class", "predicted_gas",
    "confidence", "class_probabilities", "consensus_predicted_class",
    "consensus_predicted_gas", "consensus_confidence", "consensus_probabilities",
    "consensus_window_count", "task_type", "has_concentration",
    "ppm_base_prediction", "ppm_full_prediction", "ppm_auto_output", "decision",
    "selected_calibration", "selected_policy", "risk_score", "risk_score_name", "inference_latency_ms", "observed_hz",
    "window_size", "package_name", "package_fingerprint", "dataset_profile", "device_profile",
    "model_backend", "normalization_applied", "experiment_phase", "inference_id",
    "window_start_timestamp_iso", "window_end_timestamp_iso",
    "window_start_stream_frame_index", "window_end_stream_frame_index",
    "window_connection_id", "window_recording_complete", "recording_session_id",
]


def safe_slug(text: str, fallback: str = "experiment") -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\.\u4e00-\u9fff]+", "", text, flags=re.UNICODE)
    text = text.strip("._-")
    if len(text) > 80:
        text = text[:80].rstrip("._-")
    return text or fallback


@dataclass
class ExperimentSession:
    root_dir: Path
    metadata: Dict[str, object]
    columns: list[str] = field(default_factory=lambda: list(CSV_COLUMNS_WITH_DERIVED))
    flush_every: int = 10
    session_dir: Optional[Path] = None
    csv_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    event_log_path: Optional[Path] = None
    ai_output_path: Optional[Path] = None
    _file_handle: object = field(default=None, init=False, repr=False)
    _writer: Optional[csv.DictWriter] = field(default=None, init=False, repr=False)
    _ai_handle: object = field(default=None, init=False, repr=False)
    _ai_writer: Optional[csv.DictWriter] = field(default=None, init=False, repr=False)
    _first_row_ts: Optional[float] = field(default=None, init=False, repr=False)
    rows_written: int = 0
    ai_rows_written: int = 0

    def open(self) -> Path:
        self.root_dir = Path(self.root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = safe_slug(str(self.metadata.get("experiment_name", "experiment")))
        self.session_dir = self.root_dir / f"{stamp}_{name}"
        suffix = 1
        while self.session_dir.exists():
            self.session_dir = self.root_dir / f"{stamp}_{name}_{suffix:03d}"
            suffix += 1

        try:
            self.session_dir.mkdir(parents=True, exist_ok=False)
            self.csv_path = self.session_dir / "raw.csv"
            self.meta_path = self.session_dir / "meta.json"
            self.event_log_path = self.session_dir / "event_log.csv"
            self.ai_output_path = self.session_dir / "edge_ai_predictions.csv"

            meta = dict(self.metadata)
            meta["created_at"] = datetime.now().isoformat(timespec="seconds")
            meta["csv_columns"] = self.columns
            meta["ai_output_columns"] = AI_OUTPUT_COLUMNS
            self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.event_log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp_iso", "event", "message"])
                writer.writeheader()
                writer.writerow({
                    "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                    "event": "session_created",
                    "message": str(self.session_dir),
                })

            self._file_handle = self.csv_path.open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file_handle, fieldnames=self.columns)
            self._writer.writeheader()
            self._file_handle.flush()
            self.rows_written = 0
            self.ai_rows_written = 0
            self._first_row_ts = None
            return self.session_dir
        except Exception:
            self._safe_close_handles()
            raise

    @property
    def is_open(self) -> bool:
        return self._writer is not None and self._file_handle is not None

    def write_row(self, row: Dict[str, object]) -> None:
        if not self.is_open:
            raise RuntimeError("ExperimentSession is not open; raw.csv cannot be written")
        clean = {col: row.get(col, "") for col in self.columns}
        ts = self._float(row.get("timestamp_unix"))
        if self._first_row_ts is None:
            self._first_row_ts = ts
        clean["experiment_frame_index"] = self.rows_written
        clean["experiment_elapsed_s"] = f"{max(0.0, ts - self._first_row_ts):.6f}"
        assert self._writer is not None
        self._writer.writerow(clean)
        self.rows_written += 1
        if self.flush_every <= 1 or self.rows_written % self.flush_every == 0:
            assert self._file_handle is not None
            self._file_handle.flush()

    def write_ai_result(self, result: Dict[str, object]) -> None:
        if not self.is_open or self.session_dir is None or self.ai_output_path is None:
            raise RuntimeError("ExperimentSession is not open; edge AI output cannot be written")
        if self._ai_writer is None:
            self._ai_handle = self.ai_output_path.open("w", newline="", encoding="utf-8")
            self._ai_writer = csv.DictWriter(self._ai_handle, fieldnames=AI_OUTPUT_COLUMNS)
            self._ai_writer.writeheader()
        clean = {col: result.get(col, "") for col in AI_OUTPUT_COLUMNS}
        self._ai_writer.writerow(clean)
        self.ai_rows_written += 1
        if self.flush_every <= 1 or self.ai_rows_written % self.flush_every == 0:
            self._ai_handle.flush()

    def log_event(self, event: str, message: str = "") -> None:
        if self.event_log_path is None:
            return
        with self.event_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp_iso", "event", "message"])
            writer.writerow({
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "event": str(event),
                "message": str(message),
            })

    def update_metadata(self, patch: Dict[str, object]) -> None:
        """Atomically add runtime provenance without changing the raw CSV handle."""
        if self.meta_path is None:
            return
        current: Dict[str, object] = {}
        if self.meta_path.exists():
            loaded = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                current = loaded
        clean_patch = {str(key): value for key, value in patch.items()}
        current.update(clean_patch)
        self.metadata.update(clean_patch)
        tmp_path = self.meta_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.replace(self.meta_path)

    def close(self) -> None:
        if self.event_log_path is not None:
            try:
                self.log_event(
                    "session_closed",
                    f"rows_written={self.rows_written}, ai_rows_written={self.ai_rows_written}",
                )
            except Exception:
                pass
        self._safe_close_handles()

    def _safe_close_handles(self) -> None:
        for handle_name in ["_file_handle", "_ai_handle"]:
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.flush()
                except Exception:
                    pass
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        self._writer = None
        self._ai_writer = None

    @staticmethod
    def _float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)
