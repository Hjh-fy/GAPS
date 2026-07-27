"""Small ring buffer used by the Raspberry Pi edge UI.

The buffer stores only the most recent points for plotting. CSV logging is
handled separately and can keep the full experiment history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from frame_parser_v20 import FIELD_NAMES

ADC_FIELD_NAMES = FIELD_NAMES[:16]


@dataclass
class SensorRingBuffer:
    """Fixed-length ring buffer for 16 ADC channels and environment values."""

    max_points: int = 600
    elapsed_s: deque = field(init=False)
    adc_values: Dict[str, deque] = field(init=False)
    rh: deque = field(init=False)
    temp_c: deque = field(init=False)
    uart4_ppb: deque = field(init=False)
    uart5_ppb: deque = field(init=False)

    def __post_init__(self) -> None:
        self.elapsed_s = deque(maxlen=self.max_points)
        self.adc_values = {name: deque(maxlen=self.max_points) for name in ADC_FIELD_NAMES}
        self.rh = deque(maxlen=self.max_points)
        self.temp_c = deque(maxlen=self.max_points)
        self.uart4_ppb = deque(maxlen=self.max_points)
        self.uart5_ppb = deque(maxlen=self.max_points)

    def clear(self) -> None:
        self.elapsed_s.clear()
        for buf in self.adc_values.values():
            buf.clear()
        self.rh.clear()
        self.temp_c.clear()
        self.uart4_ppb.clear()
        self.uart5_ppb.clear()

    def append_row(self, row: Dict[str, object]) -> None:
        self.elapsed_s.append(float(row.get("elapsed_s", 0.0)))
        for name in ADC_FIELD_NAMES:
            self.adc_values[name].append(float(row.get(name, 0.0)))
        self.rh.append(float(row.get("aht20_rh", 0.0)))
        self.temp_c.append(float(row.get("aht20_temp_c", 0.0)))
        self.uart4_ppb.append(float(row.get("uart4_wz_h3_nk_ppb", 0.0)))
        self.uart5_ppb.append(float(row.get("uart5_tb200b_ppb", 0.0)))

    def plot_arrays(self) -> Tuple[List[float], Dict[str, List[float]]]:
        """Return copies suitable for pyqtgraph setData()."""
        xs = list(self.elapsed_s)
        ys = {name: list(buf) for name, buf in self.adc_values.items()}
        return xs, ys

    def latest(self) -> Dict[str, float]:
        if not self.elapsed_s:
            return {}
        out = {name: float(buf[-1]) for name, buf in self.adc_values.items() if buf}
        out["elapsed_s"] = float(self.elapsed_s[-1])
        out["aht20_rh"] = float(self.rh[-1]) if self.rh else 0.0
        out["aht20_temp_c"] = float(self.temp_c[-1]) if self.temp_c else 0.0
        out["uart4_wz_h3_nk_ppb"] = float(self.uart4_ppb[-1]) if self.uart4_ppb else 0.0
        out["uart5_tb200b_ppb"] = float(self.uart5_ppb[-1]) if self.uart5_ppb else 0.0
        return out
