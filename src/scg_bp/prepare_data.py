from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import load_with_overrides
from .io_bp import iter_bp_files, read_bp_raw, standardize_bp
from .io_scg import iter_scg_files, quick_scg_meta
from .utils import ensure_dir


def parse_subject_session(path: Path, data_root: Path) -> tuple[str, str]:
    rel = path.relative_to(data_root)
    parts = rel.parts
    subject = parts[0]
    session = "default"
    if len(parts) >= 3:
        session = parts[1]
    return subject, session


def build_bp_index(data_root: Path, strict: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for bp_file in iter_bp_files(data_root):
        subject, session = parse_subject_session(bp_file, data_root)
        try:
            raw, fmt = read_bp_raw(bp_file)
            std = standardize_bp(raw, subject, session, bp_file, fmt)
            if not std.empty:
                frames.append(std)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            print(f"[WARN] Skip BP file due to parse error: {bp_file} | {type(e).__name__}: {e}")

    if not frames:
        raise RuntimeError("No BP records found after parsing.")

    bp_index = pd.concat(frames, ignore_index=True)
    bp_index["bp_row_index"] = bp_index.groupby(["subject_id", "session_id"]).cumcount()
    return bp_index


def build_signal_index(data_root: Path) -> pd.DataFrame:
    records = []
    for scg_file in iter_scg_files(data_root):
        subject, session = parse_subject_session(scg_file, data_root)
        meta = quick_scg_meta(scg_file, subject, session)
        records.append(asdict(meta))
    if not records:
        raise RuntimeError("No SCG files found.")
    return pd.DataFrame(records)


def build_sample_index(
    bp_index: pd.DataFrame,
    signal_index: pd.DataFrame,
    sample_rate_hz: int,
    window_seconds: int,
) -> pd.DataFrame:
    rows = []
    window_size = sample_rate_hz * window_seconds

    grouped_bp = bp_index.groupby(["subject_id", "session_id"], dropna=False)
    signal_map = signal_index.groupby(["subject_id", "session_id"], dropna=False)

    for (subject, session), bp_group in grouped_bp:
        candidates = None
        if (subject, session) in signal_map.groups:
            candidates = signal_map.get_group((subject, session))
        else:
            # Fallback to any signal file in the same subject.
            same_subject = signal_index[signal_index["subject_id"] == subject]
            if not same_subject.empty:
                candidates = same_subject

        if candidates is None or candidates.empty:
            continue

        chosen = candidates.sort_values("n_rows", ascending=False).iloc[0]
        n_rows = int(chosen["n_rows"])
        if n_rows < window_size:
            continue

        n_bp = len(bp_group)
        for local_idx, (_, bp_row) in enumerate(bp_group.reset_index(drop=True).iterrows()):
            ratio = (local_idx + 1) / (n_bp + 1)
            center = int(ratio * n_rows)
            start = max(0, min(center - window_size // 2, n_rows - window_size))
            end = start + window_size

            rows.append(
                {
                    "sample_id": f"{subject}_{session}_{local_idx:04d}",
                    "subject_id": subject,
                    "session_id": session,
                    "scg_file": chosen["source_file"],
                    "scg_mode": chosen["channel_mode"],
                    "start_row": start,
                    "end_row": end,
                    "window_size": window_size,
                    "bp_time_token": bp_row.get("bp_time_token", ""),
                    "SBP": float(bp_row["SBP"]),
                    "DBP": float(bp_row["DBP"]),
                    "HR": bp_row.get("HR", pd.NA),
                    "PP": bp_row.get("PP", pd.NA),
                }
            )

    if not rows:
        raise RuntimeError("No training samples generated. Check BP/SCG availability.")

    out = pd.DataFrame(rows)
    out["SBP"] = pd.to_numeric(out["SBP"], errors="coerce")
    out["DBP"] = pd.to_numeric(out["DBP"], errors="coerce")
    out = out.dropna(subset=["SBP", "DBP"]).reset_index(drop=True)
    return out


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)

    data_root = Path(cfg["paths"]["data_root"]).resolve()
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])

    sample_rate_hz = int(cfg["scg"]["sample_rate_hz"])
    window_seconds = int(cfg["window"]["seconds"])

    bp_index = build_bp_index(data_root, strict=bool(cfg.get("bp", {}).get("strict", False)))
    signal_index = build_signal_index(data_root)
    sample_index = build_sample_index(bp_index, signal_index, sample_rate_hz, window_seconds)

    bp_path = Path(cfg["output"]["bp_index"])
    sig_path = Path(cfg["output"]["signal_index"])
    smp_path = Path(cfg["output"]["sample_index"])

    ensure_dir(bp_path.parent)
    ensure_dir(sig_path.parent)
    ensure_dir(smp_path.parent)

    bp_index.to_csv(bp_path, index=False)
    signal_index.to_csv(sig_path, index=False)
    sample_index.to_csv(smp_path, index=False)

    summary = {
        "bp_records": len(bp_index),
        "signal_files": len(signal_index),
        "samples": len(sample_index),
        "processed_dir": str(processed_dir),
    }
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BP/SCG indices.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
