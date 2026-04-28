from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config root must be dict: {path}")
    return obj


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(items: list[str]) -> dict[str, Any]:
    """Parse k=v strings to nested dict using dot keys."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override (expected k=v): {item}")
        key, raw_val = item.split("=", 1)
        parts = key.split(".")
        cursor = out
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})

        val: Any = raw_val
        low = raw_val.lower()
        if low in {"true", "false"}:
            val = low == "true"
        else:
            try:
                if "." in raw_val:
                    val = float(raw_val)
                else:
                    val = int(raw_val)
            except ValueError:
                pass

        cursor[parts[-1]] = val
    return out


def load_with_overrides(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    if overrides:
        cfg = deep_update(cfg, parse_override(overrides))
    return cfg
