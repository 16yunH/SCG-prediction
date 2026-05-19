from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def parse_bp_time_token(token: Any) -> str:
    """Normalize BP time token like 61130 -> 061130."""
    if token is None:
        return ""
    text = str(token).strip()
    if not text:
        return ""
    # Remove decimal artifacts from excel numbers like 61130.0
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return ""
    return digits.zfill(6)


def bp_token_to_minutes(token: Any) -> int | None:
    """Parse a DDHHMM-style BP time token to absolute minutes within a month."""
    text = parse_bp_time_token(token)
    if len(text) != 6:
        return None
    try:
        day = int(text[:2])
        hour = int(text[2:4])
        minute = int(text[4:6])
    except ValueError:
        return None
    if hour >= 24 or minute >= 60:
        return None
    return day * 24 * 60 + hour * 60 + minute


def minutes_to_bp_token(minutes: int) -> str:
    """Format absolute minutes within a month as DDHHMM."""
    day, rem = divmod(int(round(minutes)), 24 * 60)
    hour, minute = divmod(rem, 60)
    return f"{day:02d}{hour:02d}{minute:02d}"


def pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None
