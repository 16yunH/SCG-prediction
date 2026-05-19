from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils import bp_token_to_minutes, minutes_to_bp_token


def as_int_list(raw: Any, default: list[int]) -> list[int]:
    if raw is None:
        return default
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return default


def label_group_id(row: pd.Series) -> str:
    parts = [
        str(row.get("subject_id", "")),
        str(row.get("session_id", "")),
        str(row.get("bp_row_index", "")),
        str(row.get("bp_time_token", "")),
        str(row.get("SBP", "")),
        str(row.get("DBP", "")),
        str(row.get("HR", "")),
    ]
    return "||".join(parts)


def safe_id_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)


def repair_bp_time_tokens(bp_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add parsed BP minutes and repair obvious invalid DDHHMM tokens in order."""
    out = bp_index.copy()
    out["bp_time_original"] = out["bp_time_token"].astype(str)
    out["bp_time_status"] = "valid"
    out["bp_time_correction"] = ""
    out["bp_time_minutes"] = out["bp_time_token"].map(bp_token_to_minutes)

    corrections: list[dict[str, Any]] = []
    for _, idxs in out.groupby(["subject_id", "session_id"], sort=False).groups.items():
        ordered = out.loc[list(idxs)].sort_values("bp_row_index")
        valid_minutes = [int(x) for x in ordered["bp_time_minutes"].dropna().tolist()]
        positive_gaps = [b - a for a, b in zip(valid_minutes, valid_minutes[1:]) if b > a]
        default_gap = int(round(pd.Series(positive_gaps).median())) if positive_gaps else 5

        ordered_idxs = list(ordered.index)
        for pos, (idx, row) in enumerate(ordered.iterrows()):
            if pd.notna(row["bp_time_minutes"]):
                continue

            prev_valid = out.loc[list(reversed(ordered_idxs[:pos]))]
            prev_valid = prev_valid[pd.notna(prev_valid["bp_time_minutes"])]
            next_valid = out.loc[ordered_idxs[pos + 1 :]]
            next_valid = next_valid[pd.notna(next_valid["bp_time_minutes"])]

            inferred: int | None = None
            reason = "invalid_token"
            if not prev_valid.empty and not next_valid.empty:
                inferred = int(round((float(prev_valid.iloc[0]["bp_time_minutes"]) + float(next_valid.iloc[0]["bp_time_minutes"])) / 2))
                reason = "midpoint_between_neighbors"
            elif not prev_valid.empty:
                inferred = int(float(prev_valid.iloc[0]["bp_time_minutes"])) + default_gap
                reason = "previous_plus_median_gap"
            elif not next_valid.empty:
                inferred = int(float(next_valid.iloc[0]["bp_time_minutes"])) - default_gap
                reason = "next_minus_median_gap"

            if inferred is None:
                out.loc[idx, "bp_time_status"] = "invalid_unrepaired"
                continue

            new_token = minutes_to_bp_token(inferred)
            out.loc[idx, "bp_time_token"] = new_token
            out.loc[idx, "bp_time_minutes"] = inferred
            out.loc[idx, "bp_time_status"] = "repaired"
            out.loc[idx, "bp_time_correction"] = reason
            corrections.append(
                {
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "bp_row_index": int(row["bp_row_index"]),
                    "original_token": row["bp_time_original"],
                    "corrected_token": new_token,
                    "reason": reason,
                }
            )

    out["label_group_id"] = out.apply(label_group_id, axis=1)
    return out, pd.DataFrame(corrections)


def jitter_offsets(sample_rate_hz: int, stride_seconds: int, jitter_steps: int) -> list[tuple[int, int]]:
    if jitter_steps <= 0 or stride_seconds <= 0:
        return [(0, 0)]
    offsets: list[tuple[int, int]] = []
    for step in range(-jitter_steps, jitter_steps + 1):
        offset_sec = int(step * stride_seconds)
        offset_rows = int(round(offset_sec * sample_rate_hz))
        offsets.append((offset_sec, offset_rows))
    return offsets


class WindowBuilder:
    def __init__(
        self,
        sample_rate_hz: int,
        window_seconds: int,
        alignment_method: str,
        stride_seconds: int,
        jitter_steps: int,
        window_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.window_seconds = window_seconds
        self.alignment_method = alignment_method
        self.window_cfg = window_cfg or {}
        self.measured_window_seconds = as_int_list(self.window_cfg.get("measured_seconds"), [int(window_seconds)])
        if "measured_offsets_sec" in self.window_cfg:
            self.measured_offsets_sec = as_int_list(self.window_cfg.get("measured_offsets_sec"), [0])
        else:
            self.measured_offsets_sec = [offset_sec for offset_sec, _ in jitter_offsets(sample_rate_hz, stride_seconds, jitter_steps)]
        self.measured_label_weight = float(self.window_cfg.get("measured_label_weight", 1.0))
        self.alignment = alignment_method if ("measured_offsets_sec" in self.window_cfg or jitter_steps <= 0) else f"{alignment_method}+jitter"

        interp_cfg = dict(self.window_cfg.get("interpolated", {}) or {})
        self.interpolated_enabled = bool(interp_cfg.get("enabled", False))
        self.interpolated_stride_sec = int(interp_cfg.get("stride_seconds", 30))
        self.interpolated_margin_sec = int(interp_cfg.get("exclude_margin_seconds", self.interpolated_stride_sec))
        self.interpolated_max_gap_min = float(interp_cfg.get("max_gap_minutes", 20.0))
        self.interpolated_label_weight = float(interp_cfg.get("label_weight", 0.2))

        unlabeled_cfg = dict(self.window_cfg.get("unlabeled", {}) or {})
        self.unlabeled_enabled = bool(unlabeled_cfg.get("enabled", False))
        self.unlabeled_window_seconds = as_int_list(unlabeled_cfg.get("seconds"), [max(self.measured_window_seconds)])
        self.unlabeled_stride_sec = int(unlabeled_cfg.get("stride_seconds", 30))

    def build(self, bp_index: pd.DataFrame, signal_index: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, str]]]:
        rows: list[dict[str, Any]] = []
        exclusions: list[dict[str, str]] = []
        signal_map = signal_index.groupby(["subject_id", "session_id"], dropna=False)

        def choose_signal(subject: Any, session: Any) -> tuple[pd.Series | None, str]:
            if (subject, session) in signal_map.groups:
                candidates = signal_map.get_group((subject, session))
                return candidates.sort_values("n_rows", ascending=False).iloc[0], "same_session"
            candidates = signal_index[signal_index["subject_id"] == subject]
            if candidates.empty:
                return None, "same_subject_fallback"
            return candidates.sort_values("n_rows", ascending=False).iloc[0], "same_subject_fallback"

        def center_from_time(bp_group: pd.DataFrame, local_idx: int, n_rows: int, win_size: int, time_min: float | None = None) -> int:
            if self.alignment_method in {"bp_time_interpolation", "time_interpolation"} and time_min is not None:
                times = pd.to_numeric(bp_group.get("bp_time_minutes", pd.Series(dtype=float)), errors="coerce").dropna()
                if len(times) >= 2 and float(times.max()) > float(times.min()):
                    ratio = (float(time_min) - float(times.min())) / (float(times.max()) - float(times.min()))
                    ratio = max(0.0, min(1.0, ratio))
                    return int(round((win_size / 2) + ratio * max(0, n_rows - win_size)))
            ratio = (local_idx + 1) / (len(bp_group) + 1)
            return int(ratio * n_rows)

        def add_window(**kwargs: Any) -> None:
            chosen = kwargs["chosen"]
            win_seconds = int(kwargs["win_seconds"])
            win_size = int(self.sample_rate_hz * win_seconds)
            n_rows = int(chosen.get("array_rows", chosen["n_rows"]))
            if n_rows < win_size:
                return
            start = max(0, min(int(kwargs["center"]) - win_size // 2, n_rows - win_size))
            flags = list(kwargs.get("qc_flags") or [])
            if kwargs["candidate_scope"] != "same_session":
                flags.append(kwargs["candidate_scope"])
            if kwargs["label_source"] == "measured_bp" and not kwargs.get("bp_time_token", ""):
                flags.append("missing_bp_time_token")
            if int(kwargs.get("window_offset_sec", 0)) != 0:
                flags.append(f"offset_sec={int(kwargs.get('window_offset_sec', 0))}")
            rows.append(
                {
                    "sample_id": kwargs["sample_id"],
                    "subject_id": kwargs["subject"],
                    "session_id": kwargs["session"],
                    "signal_id": chosen["signal_id"],
                    "signal_array": chosen.get("array_path", ""),
                    "scg_file": chosen["source_file"],
                    "scg_mode": chosen["channel_mode"],
                    "start_row": int(start),
                    "end_row": int(start + win_size),
                    "window_size": int(win_size),
                    "window_seconds": int(win_seconds),
                    "window_offset_sec": int(kwargs.get("window_offset_sec", 0)),
                    "bp_time_token": kwargs.get("bp_time_token", ""),
                    "bp_time_minutes": kwargs.get("bp_time_minutes", pd.NA),
                    "SBP": kwargs.get("sbp", pd.NA),
                    "DBP": kwargs.get("dbp", pd.NA),
                    "HR": kwargs.get("hr", pd.NA),
                    "PP": kwargs.get("pp", pd.NA),
                    "label_source": kwargs["label_source"],
                    "label_weight": float(kwargs["label_weight"]),
                    "is_supervised": bool(kwargs["label_source"] != "unlabeled"),
                    "label_group_id": kwargs["label_group_id"],
                    "interval_group_id": kwargs["interval_group_id"],
                    "left_label_group_id": kwargs.get("left_label_group_id", ""),
                    "right_label_group_id": kwargs.get("right_label_group_id", ""),
                    "source_bp_row_index": kwargs.get("source_bp_row_index", ""),
                    "left_bp_row_index": kwargs.get("left_bp_row_index", ""),
                    "right_bp_row_index": kwargs.get("right_bp_row_index", ""),
                    "alignment_method": self.alignment,
                    "qc_flags": ";".join(flags),
                }
            )

        for (subject, session), bp_group in bp_index.groupby(["subject_id", "session_id"], dropna=False):
            chosen, candidate_scope = choose_signal(subject, session)
            if chosen is None:
                exclusions.append({"subject_id": str(subject), "session_id": str(session), "reason": "no_signal"})
                continue
            n_rows = int(chosen.get("array_rows", chosen["n_rows"]))
            if n_rows < self.sample_rate_hz * int(self.window_seconds):
                exclusions.append({"subject_id": str(subject), "session_id": str(session), "reason": "signal_shorter_than_window"})
                continue

            bp_group = bp_group.reset_index(drop=True)
            for local_idx, bp_row in bp_group.iterrows():
                for win_seconds in self.measured_window_seconds:
                    win_size = int(self.sample_rate_hz * win_seconds)
                    base_center = center_from_time(
                        bp_group,
                        local_idx,
                        n_rows,
                        win_size,
                        float(bp_row["bp_time_minutes"]) if pd.notna(bp_row.get("bp_time_minutes", pd.NA)) else None,
                    )
                    for offset_idx, offset_sec in enumerate(self.measured_offsets_sec):
                        lg = str(bp_row.get("label_group_id", label_group_id(bp_row)))
                        add_window(
                            subject=subject,
                            session=session,
                            chosen=chosen,
                            candidate_scope=candidate_scope,
                            center=base_center + int(round(offset_sec * self.sample_rate_hz)),
                            win_seconds=int(win_seconds),
                            sample_id=f"{safe_id_part(subject)}_{safe_id_part(session)}_m{int(bp_row['bp_row_index']):04d}_s{int(win_seconds):02d}_o{offset_idx:02d}",
                            label_source="measured_bp",
                            label_weight=self.measured_label_weight,
                            sbp=float(bp_row["SBP"]),
                            dbp=float(bp_row["DBP"]),
                            hr=float(bp_row["HR"]) if pd.notna(bp_row.get("HR", pd.NA)) else None,
                            pp=float(bp_row["PP"]) if pd.notna(bp_row.get("PP", pd.NA)) else None,
                            bp_time_token=str(bp_row.get("bp_time_token", "")),
                            bp_time_minutes=float(bp_row["bp_time_minutes"]) if pd.notna(bp_row.get("bp_time_minutes", pd.NA)) else pd.NA,
                            label_group_id=lg,
                            interval_group_id=lg,
                            left_label_group_id=lg,
                            right_label_group_id=lg,
                            source_bp_row_index=int(bp_row["bp_row_index"]),
                            window_offset_sec=int(offset_sec),
                        )

            if self.interpolated_enabled and len(bp_group) >= 2:
                self._add_interpolated_windows(bp_group, chosen, candidate_scope, n_rows, center_from_time, add_window)

        if self.unlabeled_enabled:
            self._add_unlabeled_windows(signal_index, add_window)

        if not rows:
            raise RuntimeError("No training windows generated. Check BP/SCG availability.")
        out = pd.DataFrame(rows)
        out["SBP"] = pd.to_numeric(out["SBP"], errors="coerce")
        out["DBP"] = pd.to_numeric(out["DBP"], errors="coerce")
        if "is_supervised" not in out.columns:
            out["is_supervised"] = out[["SBP", "DBP"]].notna().all(axis=1)
        return out, exclusions

    def _add_interpolated_windows(self, bp_group: pd.DataFrame, chosen: pd.Series, candidate_scope: str, n_rows: int, center_from_time: Any, add_window: Any) -> None:
        subject = bp_group["subject_id"].iloc[0]
        session = bp_group["session_id"].iloc[0]
        for left_idx in range(len(bp_group) - 1):
            left = bp_group.iloc[left_idx]
            right = bp_group.iloc[left_idx + 1]
            if pd.isna(left.get("bp_time_minutes", pd.NA)) or pd.isna(right.get("bp_time_minutes", pd.NA)):
                continue
            left_min = float(left["bp_time_minutes"])
            right_min = float(right["bp_time_minutes"])
            gap_sec = int(round((right_min - left_min) * 60))
            if gap_sec <= self.interpolated_margin_sec * 2 or gap_sec > self.interpolated_max_gap_min * 60:
                continue
            left_lg = str(left.get("label_group_id", label_group_id(left)))
            right_lg = str(right.get("label_group_id", label_group_id(right)))
            interval_group_id = f"interval||{left_lg}||{right_lg}"
            t = self.interpolated_margin_sec
            interp_idx = 0
            while t < gap_sec - self.interpolated_margin_sec + 1:
                frac = t / gap_sec
                interp_min = left_min + (t / 60.0)
                for win_seconds in self.measured_window_seconds:
                    win_size = int(self.sample_rate_hz * win_seconds)
                    center = center_from_time(bp_group, left_idx, n_rows, win_size, interp_min)
                    add_window(
                        subject=subject,
                        session=session,
                        chosen=chosen,
                        candidate_scope=candidate_scope,
                        center=center,
                        win_seconds=int(win_seconds),
                        sample_id=f"{safe_id_part(subject)}_{safe_id_part(session)}_i{int(left['bp_row_index']):04d}_{int(right['bp_row_index']):04d}_t{interp_idx:03d}_s{int(win_seconds):02d}",
                        label_source="interpolated_bp",
                        label_weight=self.interpolated_label_weight,
                        sbp=float(left["SBP"]) + frac * (float(right["SBP"]) - float(left["SBP"])),
                        dbp=float(left["DBP"]) + frac * (float(right["DBP"]) - float(left["DBP"])),
                        hr=(float(left["HR"]) + frac * (float(right["HR"]) - float(left["HR"])) if pd.notna(left.get("HR", pd.NA)) and pd.notna(right.get("HR", pd.NA)) else None),
                        pp=(float(left["PP"]) + frac * (float(right["PP"]) - float(left["PP"])) if pd.notna(left.get("PP", pd.NA)) and pd.notna(right.get("PP", pd.NA)) else None),
                        bp_time_token=minutes_to_bp_token(int(round(interp_min))),
                        bp_time_minutes=interp_min,
                        label_group_id=f"interp||{interval_group_id}||{interp_idx:03d}",
                        interval_group_id=interval_group_id,
                        left_label_group_id=left_lg,
                        right_label_group_id=right_lg,
                        left_bp_row_index=int(left["bp_row_index"]),
                        right_bp_row_index=int(right["bp_row_index"]),
                        qc_flags=["pseudo_label"],
                    )
                interp_idx += 1
                t += self.interpolated_stride_sec

    def _add_unlabeled_windows(self, signal_index: pd.DataFrame, add_window: Any) -> None:
        for _, chosen in signal_index.iterrows():
            n_rows = int(chosen.get("array_rows", chosen["n_rows"]))
            subject = chosen["subject_id"]
            session = chosen["session_id"]
            for win_seconds in self.unlabeled_window_seconds:
                win_size = int(self.sample_rate_hz * win_seconds)
                stride_rows = max(1, int(round(self.unlabeled_stride_sec * self.sample_rate_hz)))
                if n_rows < win_size:
                    continue
                for idx, start in enumerate(range(0, n_rows - win_size + 1, stride_rows)):
                    add_window(
                        subject=subject,
                        session=session,
                        chosen=chosen,
                        candidate_scope="same_session",
                        center=start + win_size // 2,
                        win_seconds=int(win_seconds),
                        sample_id=f"{safe_id_part(subject)}_{safe_id_part(session)}_{safe_id_part(chosen['signal_id'])}_u{idx:06d}_s{int(win_seconds):02d}",
                        label_source="unlabeled",
                        label_weight=0.0,
                        sbp=None,
                        dbp=None,
                        hr=None,
                        pp=None,
                        bp_time_token="",
                        bp_time_minutes=None,
                        label_group_id=f"unlabeled||{chosen['signal_id']}||{idx:06d}",
                        interval_group_id=f"unlabeled||{chosen['signal_id']}||{idx:06d}",
                    )


def build_window_index(
    bp_index: pd.DataFrame,
    signal_index: pd.DataFrame,
    sample_rate_hz: int,
    window_seconds: int,
    alignment_method: str,
    stride_seconds: int,
    jitter_steps: int,
    window_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    builder = WindowBuilder(sample_rate_hz, window_seconds, alignment_method, stride_seconds, jitter_steps, window_cfg)
    return builder.build(bp_index, signal_index)
