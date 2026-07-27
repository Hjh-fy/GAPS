"""Streaming parser for the STM32 20-field gas-sensor frame protocol.

Protocol:
    0x80 0x81 + 20 little-endian uint16 values + 0x82

The parser accepts arbitrary byte chunks, handles partial frames and
resynchronizes after noise.  Framing validity and value plausibility are kept
separate: a frame can be structurally valid but contain suspicious ADC values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import struct
import time
from typing import Dict, List, Optional, Sequence, Tuple


HEAD = b"\x80\x81"
TAIL = 0x82
NUM_VALUES = 20
PAYLOAD_SIZE = NUM_VALUES * 2
FRAME_SIZE = len(HEAD) + PAYLOAD_SIZE + 1
UNPACK_FMT = "<20H"

FIELD_NAMES = [
    "adc_ch0_pa0", "adc_ch1_pa1", "adc_ch2_pa2", "adc_ch3_pa3",
    "adc_ch4_pa4", "adc_ch5_pa5", "adc_ch6_pa6", "adc_ch7_pa7",
    "adc_ch8_pb0", "adc_ch9_pb1", "adc_ch10_pc0", "adc_ch11_pc1",
    "adc_ch12_pc2", "adc_ch13_pc3", "adc_ch14_pc4", "adc_ch15_pc5",
    "aht20_rh", "aht20_temp_c", "uart4_wz_h3_nk_ppb", "uart5_tb200b_ppb",
]

RLOAD_OHM = {
    "r1_ohm": 150_000.0,
    "r2_ohm": 510_000.0,
    "r3_ohm": 15_000.0,
}
ADC_VREF = 3.3
ADC_MAX = 4095.0
ADC_EPS_V = 0.0005

# Legacy frame_index/elapsed_s are preserved.  The UI overwrites them with a
# stream-global sequence and additionally stores reconnect/experiment indices.
BASE_COLUMNS = [
    "timestamp_iso", "timestamp_unix", "elapsed_s", "frame_index",
    "stream_elapsed_s", "stream_frame_index",
    "connection_id", "connection_elapsed_s", "connection_frame_index",
    "experiment_elapsed_s", "experiment_frame_index",
    "frame_plausible", "plausibility_issue",
]
RAW_CSV_COLUMNS = BASE_COLUMNS + FIELD_NAMES
DERIVED_COLUMNS = [
    "adc_ch0_voltage_v", "adc_ch1_voltage_v", "adc_ch2_voltage_v",
    "r1_ohm", "r2_ohm", "r3_ohm",
]
CSV_COLUMNS_WITH_DERIVED = RAW_CSV_COLUMNS + DERIVED_COLUMNS


def adc_raw_to_voltage(raw: int, vref: float = ADC_VREF) -> float:
    return float(raw) * vref / ADC_MAX + ADC_EPS_V


def voltage_to_sensor_resistance(voltage: float, rload_ohm: float, vref: float = ADC_VREF) -> float:
    if not math.isfinite(voltage) or voltage <= 1e-9:
        return float("nan")
    if voltage >= vref:
        return 0.0
    return (vref - voltage) * float(rload_ohm) / voltage


@dataclass
class ParserStats:
    total_bytes: int = 0
    good_frames: int = 0
    bad_frames: int = 0
    dropped_bytes: int = 0
    resync_count: int = 0
    implausible_frames: int = 0


@dataclass
class ParsedFrame:
    frame_index: int
    timestamp_unix: float
    values: Tuple[int, ...]
    plausible: bool = True
    plausibility_issue: str = ""

    def as_raw_dict(self, start_timestamp: Optional[float] = None) -> Dict[str, object]:
        elapsed = 0.0 if start_timestamp is None else self.timestamp_unix - start_timestamp
        row: Dict[str, object] = {
            "timestamp_iso": datetime.fromtimestamp(self.timestamp_unix).isoformat(timespec="milliseconds"),
            "timestamp_unix": f"{self.timestamp_unix:.6f}",
            "elapsed_s": f"{elapsed:.6f}",
            "frame_index": self.frame_index,
            "stream_elapsed_s": "",
            "stream_frame_index": "",
            "connection_id": "",
            "connection_elapsed_s": f"{elapsed:.6f}",
            "connection_frame_index": self.frame_index,
            "experiment_elapsed_s": "",
            "experiment_frame_index": "",
            "frame_plausible": int(self.plausible),
            "plausibility_issue": self.plausibility_issue,
        }
        for name, value in zip(FIELD_NAMES, self.values):
            row[name] = int(value)
        return row

    def as_csv_dict(self, start_timestamp: Optional[float] = None, derived: bool = True) -> Dict[str, object]:
        row = self.as_raw_dict(start_timestamp=start_timestamp)
        if derived:
            add_derived_columns(row)
        return row


def add_derived_columns(row: Dict[str, object]) -> None:
    raw0 = int(row.get("adc_ch0_pa0", 0))
    raw1 = int(row.get("adc_ch1_pa1", 0))
    raw2 = int(row.get("adc_ch2_pa2", 0))
    v0, v1, v2 = adc_raw_to_voltage(raw0), adc_raw_to_voltage(raw1), adc_raw_to_voltage(raw2)
    r1 = voltage_to_sensor_resistance(v0, RLOAD_OHM["r1_ohm"])
    r2 = voltage_to_sensor_resistance(v1, RLOAD_OHM["r2_ohm"])
    r3 = voltage_to_sensor_resistance(v2, RLOAD_OHM["r3_ohm"])
    row["adc_ch0_voltage_v"] = f"{v0:.6f}"
    row["adc_ch1_voltage_v"] = f"{v1:.6f}"
    row["adc_ch2_voltage_v"] = f"{v2:.6f}"
    row["r1_ohm"] = "" if not math.isfinite(r1) else int(round(r1))
    row["r2_ohm"] = "" if not math.isfinite(r2) else int(round(r2))
    row["r3_ohm"] = "" if not math.isfinite(r3) else int(round(r3))


def check_value_plausibility(values: Sequence[int]) -> Tuple[bool, str]:
    """Soft validation only; suspicious frames are retained and annotated."""
    bad_adc = [idx for idx, value in enumerate(values[:16]) if not 0 <= int(value) <= int(ADC_MAX)]
    if bad_adc:
        return False, "adc_out_of_12bit_range:" + ",".join(str(i) for i in bad_adc)
    return True, ""


class STM32FrameParserV20:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._next_frame_index = 0
        self.stats = ParserStats()

    def reset(self) -> None:
        self._buf.clear()
        self._next_frame_index = 0
        self.stats = ParserStats()

    @property
    def buffered_bytes(self) -> int:
        return len(self._buf)

    def feed(self, data: bytes, timestamp_unix: Optional[float] = None) -> List[ParsedFrame]:
        if not data:
            return []
        if timestamp_unix is None:
            timestamp_unix = time.time()
        self.stats.total_bytes += len(data)
        self._buf.extend(data)
        frames: List[ParsedFrame] = []

        while True:
            head_pos = self._buf.find(HEAD)
            if head_pos < 0:
                if self._buf and self._buf[-1] == HEAD[0]:
                    dropped = len(self._buf) - 1
                    del self._buf[:-1]
                else:
                    dropped = len(self._buf)
                    self._buf.clear()
                self.stats.dropped_bytes += dropped
                break
            if head_pos > 0:
                del self._buf[:head_pos]
                self.stats.dropped_bytes += head_pos
                self.stats.resync_count += 1
            if len(self._buf) < FRAME_SIZE:
                break
            if self._buf[FRAME_SIZE - 1] != TAIL:
                del self._buf[0]
                self.stats.bad_frames += 1
                self.stats.resync_count += 1
                continue

            payload = bytes(self._buf[len(HEAD): len(HEAD) + PAYLOAD_SIZE])
            try:
                values = tuple(int(v) for v in struct.unpack(UNPACK_FMT, payload))
            except struct.error:
                del self._buf[0]
                self.stats.bad_frames += 1
                self.stats.resync_count += 1
                continue

            plausible, issue = check_value_plausibility(values)
            if not plausible:
                self.stats.implausible_frames += 1
            frames.append(ParsedFrame(
                frame_index=self._next_frame_index,
                timestamp_unix=timestamp_unix,
                values=values,
                plausible=plausible,
                plausibility_issue=issue,
            ))
            self._next_frame_index += 1
            self.stats.good_frames += 1
            del self._buf[:FRAME_SIZE]
        return frames


def build_frame(values: Sequence[int]) -> bytes:
    if len(values) != NUM_VALUES:
        raise ValueError(f"Expected {NUM_VALUES} values, got {len(values)}")
    for value in values:
        if not 0 <= int(value) <= 0xFFFF:
            raise ValueError(f"uint16 out of range: {value}")
    return HEAD + struct.pack(UNPACK_FMT, *[int(v) for v in values]) + bytes([TAIL])
