from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCG_CANDIDATE_NAMES = {
    "ECGSCG.csv",
    "SCGECG.csv",
    "SCG.csv",
    "SCG新(60分钟).csv",
}
NON_SCG_NAME_PARTS = ("bp", "adxl", "adpd", "summary")


@dataclass
class ScgMeta:
    subject_id: str
    session_id: str
    source_file: str
    n_rows: int
    n_cols: int
    channel_mode: str
    duration_estimate_sec: float
    selected_channels: str


def is_scg_file(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    lower = path.name.lower()
    if any(part in lower for part in NON_SCG_NAME_PARTS):
        return False
    if path.name in SCG_CANDIDATE_NAMES:
        return True
    if "（" in path.name and "）" in path.name:
        return True
    return False


def iter_scg_files(data_root: Path) -> Iterable[Path]:
    for p in data_root.rglob("*.csv"):
        if is_scg_file(p):
            yield p


def select_scg_columns(columns: list[str], mode: str, input_channels: int = 6) -> list[str]:
    if mode == "9col":
        keep = [c for c in ["I3", "I4", "I5", "I6", "I7", "I8"] if c in columns]
    else:
        keep = [c for c in ["I1", "I2", "I3", "I4", "I5", "I6"] if c in columns]
    if len(keep) < input_channels:
        keep = [c for c in columns if c != "I0"][-max(input_channels, 1) :]
    return keep[:input_channels]


def quick_scg_meta(path: Path, subject_id: str, session_id: str, sample_rate_hz: int = 50) -> ScgMeta:
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

    selected = select_scg_columns(header, mode, 6) if mode != "unknown" else []
    return ScgMeta(
        subject_id=subject_id,
        session_id=session_id,
        source_file=str(path),
        n_rows=n_rows,
        n_cols=n_cols,
        channel_mode=mode,
        duration_estimate_sec=float(n_rows) / float(sample_rate_hz) if sample_rate_hz > 0 else 0.0,
        selected_channels=",".join(selected),
    )


def read_scg_window(path: Path, start_row: int, end_row: int, mode: str, input_channels: int = 6) -> pd.DataFrame:
    nrows = max(0, end_row - start_row)
    if nrows <= 0:
        return pd.DataFrame()
    df = pd.read_csv(path, skiprows=range(1, start_row + 1), nrows=nrows)
    keep = select_scg_columns(list(df.columns), mode, input_channels)
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    out = out.ffill().bfill().fillna(0.0)
    while out.shape[1] < input_channels:
        out[f"pad_{out.shape[1]}"] = 0.0
    return out.iloc[:, :input_channels]


def read_scg_full_array(path: Path, mode: str, input_channels: int = 6) -> np.ndarray:
    df = pd.read_csv(path)
    keep = select_scg_columns(list(df.columns), mode, input_channels)
    out = df[keep].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    arr = out.to_numpy(dtype=np.float32, copy=False)
    if arr.shape[1] < input_channels:
        pad = np.zeros((arr.shape[0], input_channels - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    return arr[:, :input_channels]


def write_scg_array(path: Path, mode: str, output_path: Path, input_channels: int = 6) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = read_scg_full_array(path, mode, input_channels=input_channels)
    np.save(output_path, arr.astype(np.float32, copy=False))
    return {"array_path": str(output_path), "array_rows": int(arr.shape[0]), "array_cols": int(arr.shape[1])}
