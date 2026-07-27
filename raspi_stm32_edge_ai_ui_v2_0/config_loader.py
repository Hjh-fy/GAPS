"""Small optional YAML/JSON configuration loader for the edge UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required for YAML config files") from exc
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/object")
    return data


def ui_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    serial = dict(config.get("serial") or {})
    ui = dict(config.get("ui") or {})
    logging = dict(config.get("logging") or {})
    ai = dict(config.get("ai") or {})
    out: Dict[str, Any] = {}
    mapping = {
        "port": serial.get("port"),
        "baudrate": serial.get("baudrate"),
        "fullscreen": ui.get("fullscreen"),
        "font_scale": ui.get("font_scale"),
        "max_plot_points": ui.get("max_plot_points"),
        "max_segment_rows": ui.get("max_segment_rows"),
        "data_root": logging.get("data_root"),
        "ai_package": ai.get("package"),
    }
    for key, value in mapping.items():
        if value is not None:
            out[key] = value
    return out
