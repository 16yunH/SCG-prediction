from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SCG_CANDIDATE_NAMES = {
    "ECGSCG.csv",
    "SCGECG.csv",
    "SCG.csv",
    "SCG新(60分钟).csv",
}


@dataclass
class ScgMeta:
    subject_id: str
    session_id: str
    source_file: str
    n_rows: int
    n_cols: int
    channel_mode: str


def is_scg_file(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    name = path.name
    if name in SCG_CANDIDATE_NAMES:
        return True
    # include legacy named files like 054（1）.csv / 055（6分钟）.csv
    if "（" in name and "）" in name and name.lower().endswith(".csv"):
        return True
    return False


def iter_scg_files(data_root: Path) -> Iterable[Path]:
    for p in data_root.rglob("*.csv"):
        if is_scg_file(p):
            yield p


def quick_scg_meta(path: Path, subject_id: str, session_id: str) -> ScgMeta:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split(",")
        n_cols = len(header)
        n_rows = sum(1 for _ in f)

    if n_cols >= 9:
        mode = "9col"
    elif n_cols >= 7:
        mode = "7col"
    else:
        mode = "unknown"

    return ScgMeta(
        subject_id=subject_id,
        session_id=session_id,
        source_file=str(path),
        n_rows=n_rows,
        n_cols=n_cols,
        channel_mode=mode,
    )


def read_scg_window(path: Path, start_row: int, end_row: int, mode: str) -> pd.DataFrame:
    # start_row/end_row are 0-based on data rows (excluding header)
    nrows = max(0, end_row - start_row)
    if nrows <= 0:
        return pd.DataFrame()
    df = pd.read_csv(path, skiprows=range(1, start_row + 1), nrows=nrows)

    cols = list(df.columns)
    if mode == "9col":
        keep = [c for c in ["I3", "I4", "I5", "I6", "I7", "I8"] if c in cols]
    else:
        keep = [c for c in ["I1", "I2", "I3", "I4", "I5", "I6"] if c in cols]

    if len(keep) < 6:
        # fallback to last 6 cols excluding timestamp-like I0
        keep = [c for c in cols if c != "I0"][-6:]

    out = df[keep].apply(pd.to_numeric, errors="coerce")
    out = out.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    return out
