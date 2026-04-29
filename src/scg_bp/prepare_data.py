from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_with_overrides
from .io_bp import iter_bp_files, read_bp_raw, standardize_bp
from .io_scg import iter_scg_files, quick_scg_meta, write_scg_array
from .utils import ensure_dir, save_json


def parse_subject_session(path: Path, data_root: Path) -> tuple[str, str]:
    rel = path.relative_to(data_root)
    parts = rel.parts
    subject = parts[0] if len(parts) >= 2 else "__root__"
    session = parts[1] if len(parts) >= 3 else "default"
    return subject, session


def build_raw_manifest(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bp_files = {p.resolve() for p in iter_bp_files(data_root)}
    scg_files = {p.resolve() for p in iter_scg_files(data_root)}
    for p in sorted([x for x in data_root.rglob("*") if x.is_file()]):
        subject, session = parse_subject_session(p, data_root)
        rp = p.resolve()
        role = "other"
        if rp in bp_files:
            role = "bp"
        elif rp in scg_files:
            role = "scg"
        rows.append(
            {
                "subject_id": subject,
                "session_id": session,
                "path": str(p),
                "extension": p.suffix.lower(),
                "size_bytes": int(p.stat().st_size),
                "role": role,
                "parser_status": "pending" if role in {"bp", "scg"} else "ignored",
            }
        )
    return pd.DataFrame(rows)


def build_bp_index(data_root: Path, strict: bool = False) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for bp_file in iter_bp_files(data_root):
        subject, session = parse_subject_session(bp_file, data_root)
        try:
            raw, fmt = read_bp_raw(bp_file)
            std = standardize_bp(raw, subject, session, bp_file, fmt)
            if not std.empty:
                std = std.copy()
                std["parse_method"] = fmt
                frames.append(std)
        except Exception as e:  # noqa: BLE001
            failures.append({"path": str(bp_file), "error": f"{type(e).__name__}: {e}"})
            if strict:
                raise
            print(f"[WARN] Skip BP file due to parse error: {bp_file} | {type(e).__name__}: {e}", flush=True)

    if not frames:
        raise RuntimeError("No BP records found after parsing.")

    bp_index = pd.concat(frames, ignore_index=True)
    bp_index["bp_row_index"] = bp_index.groupby(["subject_id", "session_id"]).cumcount()
    keep = [
        "subject_id",
        "session_id",
        "bp_row_index",
        "bp_time_token",
        "SBP",
        "DBP",
        "HR",
        "PP",
        "source_file",
        "format_type",
        "parse_method",
    ]
    return bp_index[[c for c in keep if c in bp_index.columns]], failures


def _safe_array_name(subject: str, session: str, idx: int) -> str:
    text = f"{idx:03d}_{subject}_{session}".replace("/", "_").replace("\\", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text) + ".npy"


def _jitter_offsets(sample_rate_hz: int, stride_seconds: int, jitter_steps: int) -> list[tuple[int, int]]:
    if jitter_steps <= 0 or stride_seconds <= 0:
        return [(0, 0)]
    offsets: list[tuple[int, int]] = []
    for step in range(-jitter_steps, jitter_steps + 1):
        offset_sec = int(step * stride_seconds)
        offset_rows = int(round(offset_sec * sample_rate_hz))
        offsets.append((offset_sec, offset_rows))
    return offsets


def build_signal_index(data_root: Path, arrays_dir: Path, sample_rate_hz: int, input_channels: int, materialize_arrays: bool) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for idx, scg_file in enumerate(iter_scg_files(data_root), start=1):
        subject, session = parse_subject_session(scg_file, data_root)
        try:
            meta = asdict(quick_scg_meta(scg_file, subject, session, sample_rate_hz=sample_rate_hz))
            meta["signal_id"] = f"sig_{idx:03d}"
            if materialize_arrays and meta["channel_mode"] != "unknown":
                array_path = arrays_dir / _safe_array_name(subject, session, idx)
                meta.update(write_scg_array(scg_file, str(meta["channel_mode"]), array_path, input_channels=input_channels))
            else:
                meta["array_path"] = ""
                meta["array_rows"] = meta["n_rows"]
                meta["array_cols"] = input_channels
            records.append(meta)
        except Exception as e:  # noqa: BLE001
            failures.append({"path": str(scg_file), "error": f"{type(e).__name__}: {e}"})
            print(f"[WARN] Skip SCG file due to parse error: {scg_file} | {type(e).__name__}: {e}", flush=True)
    if not records:
        raise RuntimeError("No SCG files found.")
    out = pd.DataFrame(records)
    cols = ["signal_id", "subject_id", "session_id", "source_file", "array_path", "n_rows", "n_cols", "channel_mode", "duration_estimate_sec", "selected_channels", "array_rows", "array_cols"]
    return out[[c for c in cols if c in out.columns]], failures


def build_window_index(
    bp_index: pd.DataFrame,
    signal_index: pd.DataFrame,
    sample_rate_hz: int,
    window_seconds: int,
    alignment_method: str,
    stride_seconds: int,
    jitter_steps: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    window_size = sample_rate_hz * window_seconds
    offsets = _jitter_offsets(sample_rate_hz, stride_seconds, jitter_steps)
    alignment = alignment_method if jitter_steps <= 0 else f"{alignment_method}+jitter"
    grouped_bp = bp_index.groupby(["subject_id", "session_id"], dropna=False)
    signal_map = signal_index.groupby(["subject_id", "session_id"], dropna=False)

    for (subject, session), bp_group in grouped_bp:
        if (subject, session) in signal_map.groups:
            candidates = signal_map.get_group((subject, session))
            candidate_scope = "same_session"
        else:
            candidates = signal_index[signal_index["subject_id"] == subject]
            candidate_scope = "same_subject_fallback"

        if candidates.empty:
            exclusions.append({"subject_id": str(subject), "session_id": str(session), "reason": "no_signal"})
            continue

        chosen = candidates.sort_values("n_rows", ascending=False).iloc[0]
        n_rows = int(chosen.get("array_rows", chosen["n_rows"]))
        if n_rows < window_size:
            exclusions.append({"subject_id": str(subject), "session_id": str(session), "reason": "signal_shorter_than_window"})
            continue

        bp_group = bp_group.reset_index(drop=True)
        n_bp = len(bp_group)
        for local_idx, bp_row in bp_group.iterrows():
            ratio = (local_idx + 1) / (n_bp + 1)
            center = int(ratio * n_rows)
            seen: set[tuple[int, int]] = set()
            for offset_idx, (offset_sec, offset_rows) in enumerate(offsets):
                offset_center = center + offset_rows
                start = max(0, min(offset_center - window_size // 2, n_rows - window_size))
                end = start + window_size
                key = (int(start), int(end))
                if key in seen:
                    continue
                seen.add(key)

                qc_flags = [] if candidate_scope == "same_session" else [candidate_scope]
                if not str(bp_row.get("bp_time_token", "")):
                    qc_flags.append("missing_bp_time_token")
                if offset_sec != 0:
                    qc_flags.append(f"jitter_sec={offset_sec}")
                rows.append(
                    {
                        "sample_id": f"{subject}_{session}_{local_idx:04d}_w{offset_idx:02d}",
                        "subject_id": subject,
                        "session_id": session,
                        "signal_id": chosen["signal_id"],
                        "signal_array": chosen.get("array_path", ""),
                        "scg_file": chosen["source_file"],
                        "scg_mode": chosen["channel_mode"],
                        "start_row": int(start),
                        "end_row": int(end),
                        "window_size": int(window_size),
                        "window_offset_sec": int(offset_sec),
                        "bp_time_token": bp_row.get("bp_time_token", ""),
                        "SBP": float(bp_row["SBP"]),
                        "DBP": float(bp_row["DBP"]),
                        "HR": bp_row.get("HR", pd.NA),
                        "PP": bp_row.get("PP", pd.NA),
                        "alignment_method": alignment,
                        "qc_flags": ";".join(qc_flags),
                    }
                )

    if not rows:
        raise RuntimeError("No training windows generated. Check BP/SCG availability.")
    out = pd.DataFrame(rows)
    out["SBP"] = pd.to_numeric(out["SBP"], errors="coerce")
    out["DBP"] = pd.to_numeric(out["DBP"], errors="coerce")
    out = out.dropna(subset=["SBP", "DBP"]).reset_index(drop=True)
    return out, exclusions


def _missing_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out[col] = float(vals.isna().mean()) if len(vals) > 0 else float("nan")
    return out


def _apply_manifest_status(
    raw_manifest: pd.DataFrame,
    bp_index: pd.DataFrame,
    signal_index: pd.DataFrame,
    bp_failures: list[dict[str, str]],
    scg_failures: list[dict[str, str]],
) -> pd.DataFrame:
    if raw_manifest.empty:
        return raw_manifest
    out = raw_manifest.copy()
    bp_ok = set(bp_index.get("source_file", pd.Series(dtype=str)).astype(str).tolist()) if not bp_index.empty else set()
    scg_ok = set(signal_index.get("source_file", pd.Series(dtype=str)).astype(str).tolist()) if not signal_index.empty else set()
    bp_fail = {str(item.get("path", "")) for item in bp_failures}
    scg_fail = {str(item.get("path", "")) for item in scg_failures}

    def _status(row: pd.Series) -> str:
        role = str(row.get("role", ""))
        path = str(row.get("path", ""))
        if role == "bp":
            if path in bp_fail:
                return "failed"
            if path in bp_ok:
                return "parsed"
            return "skipped"
        if role == "scg":
            if path in scg_fail:
                return "failed"
            if path in scg_ok:
                return "parsed"
            return "skipped"
        return "ignored"

    out["parser_status"] = out.apply(_status, axis=1)
    return out


def build_qc_report(
    raw_manifest: pd.DataFrame,
    bp_index: pd.DataFrame,
    signal_index: pd.DataFrame,
    window_index: pd.DataFrame,
    failures: dict[str, Any],
    exclusions: list[dict[str, str]],
    window_seconds: int,
    alignment_method: str,
    stride_seconds: int,
    jitter_steps: int,
) -> dict[str, Any]:
    subjects = sorted(s for s in raw_manifest["subject_id"].dropna().astype(str).unique().tolist() if s != "__root__") if not raw_manifest.empty else []
    exclusions_map: dict[str, set[str]] = {}
    for ex in exclusions:
        subject = str(ex.get("subject_id", ""))
        reason = str(ex.get("reason", ""))
        if subject and reason:
            exclusions_map.setdefault(subject, set()).add(reason)
    subject_rows = []
    for subject in subjects:
        bp_sub = bp_index[bp_index["subject_id"].astype(str) == subject] if not bp_index.empty else pd.DataFrame()
        missing = _missing_rates(bp_sub, ["SBP", "DBP", "HR", "PP"]) if not bp_sub.empty else {}
        subject_rows.append(
            {
                "subject_id": subject,
                "raw_files": int((raw_manifest["subject_id"].astype(str) == subject).sum()),
                "bp_records": int(len(bp_sub)) if not bp_index.empty else 0,
                "bp_missing_rate_sbp": missing.get("SBP", float("nan")),
                "bp_missing_rate_dbp": missing.get("DBP", float("nan")),
                "bp_missing_rate_hr": missing.get("HR", float("nan")),
                "bp_missing_rate_pp": missing.get("PP", float("nan")),
                "signal_files": int((signal_index["subject_id"].astype(str) == subject).sum()) if not signal_index.empty else 0,
                "windows": int((window_index["subject_id"].astype(str) == subject).sum()) if not window_index.empty else 0,
                "excluded_reasons": ";".join(sorted(exclusions_map.get(subject, set()))),
            }
        )
    window_size = int(window_index["window_size"].iloc[0]) if not window_index.empty and "window_size" in window_index.columns else int(window_seconds)
    return {
        "raw_files": int(len(raw_manifest)),
        "subjects_scanned": int(len(subjects)),
        "structured_bp_subjects": int(bp_index["subject_id"].nunique()) if not bp_index.empty else 0,
        "signal_subjects": int(signal_index["subject_id"].nunique()) if not signal_index.empty else 0,
        "bp_records": int(len(bp_index)),
        "signal_files": int(len(signal_index)),
        "windows": int(len(window_index)),
        "window_seconds": int(window_seconds),
        "window_size": int(window_size),
        "alignment_method": alignment_method,
        "window_stride_seconds": int(stride_seconds),
        "window_jitter_steps": int(jitter_steps),
        "excluded_subjects_note": "057a(BP) is excluded by default unless structured BP labels are provided.",
        "failures": failures,
        "exclusions": exclusions,
        "trainable_samples": int(len(window_index)),
        "by_subject": subject_rows,
    }


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    data_root = Path(cfg["paths"]["data_root"]).resolve()
    processed_dir = ensure_dir(cfg["paths"]["processed_dir"])
    arrays_dir = ensure_dir(cfg.get("paths", {}).get("arrays_dir", str(processed_dir / "arrays")))
    sample_rate_hz = int(cfg["scg"]["sample_rate_hz"])
    input_channels = int(cfg.get("scg", {}).get("input_channels", 6))
    window_seconds = int(cfg["window"]["seconds"])
    stride_seconds = int(cfg.get("window", {}).get("stride_seconds", 0))
    jitter_steps = int(cfg.get("window", {}).get("jitter_steps", 0))
    alignment_method = str(cfg.get("window", {}).get("alignment_method", "rank_interpolation"))
    materialize_arrays = bool(cfg.get("scg", {}).get("materialize_arrays", True))

    raw_manifest = build_raw_manifest(data_root)
    bp_index, bp_failures = build_bp_index(data_root, strict=bool(cfg.get("bp", {}).get("strict", False)))
    signal_index, scg_failures = build_signal_index(data_root, arrays_dir, sample_rate_hz, input_channels, materialize_arrays)
    window_index, exclusions = build_window_index(
        bp_index,
        signal_index,
        sample_rate_hz,
        window_seconds,
        alignment_method,
        stride_seconds,
        jitter_steps,
    )
    raw_manifest = _apply_manifest_status(raw_manifest, bp_index, signal_index, bp_failures, scg_failures)

    out_cfg = cfg["output"]
    paths = {
        "raw_manifest": Path(out_cfg.get("raw_manifest", processed_dir / "raw_manifest.csv")),
        "bp_index": Path(out_cfg["bp_index"]),
        "signal_index": Path(out_cfg["signal_index"]),
        "window_index": Path(out_cfg.get("window_index", out_cfg.get("sample_index", processed_dir / "window_index.csv"))),
        "sample_index": Path(out_cfg.get("sample_index", processed_dir / "sample_index.csv")),
        "qc_report": Path(out_cfg.get("qc_report", processed_dir / "qc_report.json")),
    }
    for p in paths.values():
        ensure_dir(p.parent)

    raw_manifest.to_csv(paths["raw_manifest"], index=False)
    bp_index.to_csv(paths["bp_index"], index=False)
    signal_index.to_csv(paths["signal_index"], index=False)
    window_index.to_csv(paths["window_index"], index=False)
    # Backward-compatible alias for existing split/training commands.
    window_index.to_csv(paths["sample_index"], index=False)

    qc = build_qc_report(
        raw_manifest,
        bp_index,
        signal_index,
        window_index,
        {"bp": bp_failures, "scg": scg_failures},
        exclusions,
        window_seconds,
        alignment_method,
        stride_seconds,
        jitter_steps,
    )
    save_json(paths["qc_report"], qc)
    print(
        {
            "bp_records": len(bp_index),
            "signal_files": len(signal_index),
            "windows": len(window_index),
            "processed_dir": str(processed_dir),
            "arrays_dir": str(arrays_dir),
            "qc_report": str(paths["qc_report"]),
        },
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare structured BP/SCG training assets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
