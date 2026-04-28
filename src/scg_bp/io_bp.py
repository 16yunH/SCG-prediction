from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import pandas as pd

from .utils import parse_bp_time_token, pick_column


BP_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def detect_bp_format(path: Path) -> str:
    sig = path.read_bytes()[:4]
    if sig == b"PK\x03\x04":
        return "xlsx"
    if sig == b"\xD0\xCF\x11\xE0":
        return "xls"
    return "text_csv"


def _read_xlsx_any(path: Path) -> pd.DataFrame:
    blob = path.read_bytes()
    bio = BytesIO(blob)
    return pd.read_excel(bio, sheet_name=0)


def _read_xls(path: Path) -> pd.DataFrame:
    # Prefer calamine first (faster, no C ext headaches), fallback to xlrd.
    errors: list[str] = []
    for engine in ("calamine", "xlrd", None):
        try:
            if engine is None:
                return pd.read_excel(path, sheet_name=0)
            return pd.read_excel(path, sheet_name=0, engine=engine)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{engine}: {type(e).__name__}: {e}")
    joined = " | ".join(errors)
    raise RuntimeError(
        "Failed to parse legacy XLS. Install `python-calamine` or `xlrd`. "
        f"File={path}. Errors={joined}"
    )


def _read_text_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "gb18030", "gbk", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Failed to parse text CSV with fallback encodings: {path}")


def read_bp_raw(path: Path) -> tuple[pd.DataFrame, str]:
    fmt = detect_bp_format(path)
    if fmt == "xlsx":
        return _read_xlsx_any(path), fmt
    if fmt == "xls":
        return _read_xls(path), fmt
    return _read_text_csv(path), fmt


def standardize_bp(df: pd.DataFrame, subject_id: str, session_id: str, source_file: Path, fmt: str) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    sbp_col = pick_column(cols, ["SBP", "收缩压"])
    dbp_col = pick_column(cols, ["DBP", "舒张压"])
    hr_col = pick_column(cols, ["HR", "心率"])
    pp_col = pick_column(cols, ["PP", "脉压差"])
    t_col = pick_column(cols, ["t（日时分）", "t", "time", "tʱ֣"])

    if sbp_col is None or dbp_col is None:
        raise ValueError(f"BP columns missing SBP/DBP in {source_file}")

    out = pd.DataFrame()
    out["subject_id"] = [subject_id] * len(df)
    out["session_id"] = [session_id] * len(df)
    out["source_file"] = [str(source_file)] * len(df)
    out["format_type"] = [fmt] * len(df)

    if t_col is not None:
        out["bp_time_token"] = df[t_col].map(parse_bp_time_token)
    else:
        out["bp_time_token"] = ""

    out["SBP"] = pd.to_numeric(df[sbp_col], errors="coerce")
    out["DBP"] = pd.to_numeric(df[dbp_col], errors="coerce")
    out["HR"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else pd.NA
    out["PP"] = pd.to_numeric(df[pp_col], errors="coerce") if pp_col else pd.NA

    out = out.dropna(subset=["SBP", "DBP"]).reset_index(drop=True)
    return out


def iter_bp_files(data_root: Path) -> Iterable[Path]:
    for p in data_root.rglob("*"):
        if not p.is_file():
            continue
        if not p.name.upper().startswith("BP."):
            continue
        if p.suffix.lower() not in BP_EXTENSIONS:
            continue
        yield p
