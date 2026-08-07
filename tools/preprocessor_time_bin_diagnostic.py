"""Read-only real-time 100-ms bin preprocessing diagnostic.

This module deliberately does not change the production time-aware preprocessor.
It aggregates real observations in bins and marks unfilled windows invalid; the
optional short-gap policy fills only one missing bin between finite neighbours.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any

import numpy as np


def _timeaware():
    # This audit lives in a git worktree; production preprocessing is retained
    # beside the workspace root rather than copied into the worktree.
    path = Path(__file__).resolve().parents[3] / "preprocessor_time_aware.py"
    spec = importlib.util.spec_from_file_location("audit_time_aware", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TA = _timeaware()


def aggregate_100ms(filepath: str | Path, statistic: str = "mean", short_gap: bool = False) -> dict[str, Any]:
    """Return an observation-only 10-Hz trace and transparent missingness metadata."""
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be mean or median")
    config = TA.TimeAwareConfig()
    raw_t, raw_x = TA.load_raw_data(filepath)
    t, x, clean = TA.clean_time_axis(raw_t, raw_x)
    start = np.ceil(float(t[0]) * 10.0) / 10.0
    end = np.floor(float(t[-1]) * 10.0) / 10.0
    grid = np.arange(start, end + 0.0500001, 0.1, dtype=float)
    # Round defensively so floating point bin edges do not change membership.
    bid = np.floor((t - start + 1e-9) * 10.0).astype(int)
    valid = (bid >= 0) & (bid < len(grid))
    values = np.full((len(grid), x.shape[1]), np.nan, dtype=float)
    counts = np.zeros(len(grid), dtype=int)
    for i in np.unique(bid[valid]):
        block = x[bid == i]
        counts[i] = len(block)
        values[i] = np.mean(block, axis=0) if statistic == "mean" else np.median(block, axis=0)
    missing = counts == 0
    filled = np.zeros(len(grid), dtype=bool)
    if short_gap:
        for i in range(1, len(grid) - 1):
            if missing[i] and not missing[i - 1] and not missing[i + 1]:
                values[i] = (values[i - 1] + values[i + 1]) / 2.0
                filled[i] = True
    raw_missing = missing.copy()
    # Same production baseline semantics; NaNs explicitly propagate to invalid windows.
    g = 1.0 / np.clip(values, 1e-10, None)
    base = (grid >= 20.0) & (grid < 50.0)
    g0 = np.nanmean(g[base], axis=0)
    rel = (g - g0) / g0
    crop = (grid >= 60.0) & (grid <= 170.0 + 1e-9)
    windows, metadata = [], []
    idx = np.where(crop)[0]
    for s in range(int(idx[0]), int(idx[-1]) - config.window_size + 2, config.step_size):
        take = np.arange(s, s + config.window_size)
        windows.append(rel[take].astype(np.float32))
        metadata.append({"window_start_s": float(grid[take[0]]), "window_end_s": float(grid[take[-1]] + .1),
                         "valid": bool(np.isfinite(rel[take]).all()), "empty_bin_ratio": float(np.mean(raw_missing[take])),
                         "short_gap_interpolated_ratio": float(np.mean(filled[take])),
                         "max_missing_run": int(_max_run(raw_missing[take]))})
    return {"time_s": grid, "sensors": values, "conductance": g, "baseline": g0, "relative": rel,
            "windows": np.asarray(windows, dtype=np.float32), "metadata": metadata,
            "samples_per_bin": counts, "empty_bin_ratio": float(np.mean(raw_missing)),
            "short_gap_interpolated_ratio": float(np.mean(filled)), "max_missing_run": int(_max_run(raw_missing)),
            "duplicate_timestamps": int(clean["duplicate_timestamps"])}


def _max_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest
