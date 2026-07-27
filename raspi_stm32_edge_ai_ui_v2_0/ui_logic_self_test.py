#!/usr/bin/env python3
"""Headless self-test for parser + ring buffer + experiment CSV writer.

This does not launch the Qt UI, so it can run in CI or on a headless machine.
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from csv_writer import ExperimentSession
from data_buffer import SensorRingBuffer
from frame_parser_v20 import STM32FrameParserV20, build_frame


def main() -> int:
    parser = STM32FrameParserV20()
    payload = b"noise" + build_frame(list(range(20))) + build_frame([100 + i for i in range(20)])
    frames = parser.feed(payload, timestamp_unix=1_700_000_000.0)
    assert len(frames) == 2, len(frames)
    assert parser.stats.good_frames == 2
    assert parser.stats.dropped_bytes >= 5

    first_ts = frames[0].timestamp_unix
    rows = [frame.as_csv_dict(start_timestamp=first_ts, derived=True) for frame in frames]

    buffer = SensorRingBuffer(max_points=10)
    for row in rows:
        buffer.append_row(row)
    latest = buffer.latest()
    assert latest["adc_ch0_pa0"] == 100.0
    assert latest["aht20_rh"] == 116.0
    assert latest["aht20_temp_c"] == 117.0

    with tempfile.TemporaryDirectory() as tmp:
        session = ExperimentSession(Path(tmp), {"experiment_name": "self_test", "gas_type": "air"})
        session_dir = session.open()
        for row in rows:
            session.write_row(row)
        session.update_metadata(
            {"edge_ai_runtime": {"package_fingerprint": "abc123"}}
        )
        session.close()
        csv_path = session_dir / "raw.csv"
        meta_path = session_dir / "meta.json"
        assert csv_path.exists()
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["edge_ai_runtime"]["package_fingerprint"] == "abc123"
        with csv_path.open(newline="", encoding="utf-8") as f:
            saved = list(csv.DictReader(f))
        assert len(saved) == 2
        assert saved[-1]["adc_ch15_pc5"] == "115"
        assert saved[-1]["aht20_rh"] == "116"
        assert saved[-1]["aht20_temp_c"] == "117"
        assert saved[0]["experiment_frame_index"] == "0"
        assert saved[1]["experiment_frame_index"] == "1"
        assert float(saved[0]["experiment_elapsed_s"]) == 0.0

    print("UI logic self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
