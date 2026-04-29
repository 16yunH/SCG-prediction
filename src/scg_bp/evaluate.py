from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    data["metrics_path"] = str(path)
    data["run_dir"] = data.get("run_dir", str(path.parent))
    data["run_name"] = Path(data["run_dir"]).name
    return data


def _mean_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _std_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")


def _as_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _include_metrics(metrics_path: Path, cfg: dict[str, Any]) -> bool:
    input_cfg = cfg.get("input", {})
    run_name = metrics_path.parent.name
    prefixes = _as_list(input_cfg.get("run_prefix", ""))
    if prefixes and not any(run_name.startswith(prefix) for prefix in prefixes):
        return False
    contains = _as_list(input_cfg.get("run_contains", ""))
    if contains and not any(part in run_name for part in contains):
        return False
    modes = set(_as_list(input_cfg.get("mode", "")))
    if modes:
        try:
            with metrics_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if str(data.get("mode", "all")) not in modes:
                return False
        except Exception:
            return False
    return True


def _aggregate_model(model: str, g: pd.DataFrame, fold_df: pd.DataFrame) -> dict[str, Any]:
    fold_g = fold_df[fold_df["model"] == model] if not fold_df.empty and "model" in fold_df.columns else pd.DataFrame()
    source = fold_g if not fold_g.empty else g
    row = {
        "model": model,
        "runs": int(len(g)),
        "cv_folds": int(source["fold"].nunique()) if "fold" in source.columns else int(g["folds"].sum()) if "folds" in g.columns else 0,
        "cv_val_mae_sbp_mean": _mean_numeric(source["val_mae_sbp"] if "val_mae_sbp" in source.columns else g.get("cv_val_mae_sbp", pd.Series(dtype=float))),
        "cv_val_mae_dbp_mean": _mean_numeric(source["val_mae_dbp"] if "val_mae_dbp" in source.columns else g.get("cv_val_mae_dbp", pd.Series(dtype=float))),
        "cv_val_mae_sbp_std": _std_numeric(source["val_mae_sbp"] if "val_mae_sbp" in source.columns else g.get("cv_val_mae_sbp", pd.Series(dtype=float))),
        "cv_val_mae_dbp_std": _std_numeric(source["val_mae_dbp"] if "val_mae_dbp" in source.columns else g.get("cv_val_mae_dbp", pd.Series(dtype=float))),
        "test_mae_sbp": _mean_numeric(g.get("test_mae_sbp", pd.Series(dtype=float))),
        "test_mae_dbp": _mean_numeric(g.get("test_mae_dbp", pd.Series(dtype=float))),
        "runtime_seconds_total": float(pd.to_numeric(g.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna().sum()) if "runtime_seconds" in g else float("nan"),
    }
    row["cv_val_mae_mean"] = float(np.nanmean([row["cv_val_mae_sbp_mean"], row["cv_val_mae_dbp_mean"]]))
    row["test_mae_mean"] = float(np.nanmean([row["test_mae_sbp"], row["test_mae_dbp"]]))
    return row


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    runs_dir = Path(cfg["input"]["runs_dir"])
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    summaries: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    scanned = 0
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        scanned += 1
        if not _include_metrics(metrics_path, cfg):
            continue
        summaries.append(_load_metrics(metrics_path))
        fold_path = metrics_path.parent / "fold_metrics.csv"
        if fold_path.exists():
            df = pd.read_csv(fold_path)
            df["metrics_path"] = str(metrics_path)
            df["run_dir"] = str(metrics_path.parent)
            df["run_name"] = metrics_path.parent.name
            fold_frames.append(df)

    if not summaries:
        filters = {k: v for k, v in cfg.get("input", {}).items() if k != "runs_dir"}
        raise RuntimeError(f"No metrics.json matched filters under {runs_dir}. scanned={scanned}, filters={filters}")

    raw_summary_df = pd.DataFrame(summaries)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    model_rows = [_aggregate_model(model, g, fold_df) for model, g in raw_summary_df.groupby("model", dropna=False)]
    model_summary_df = pd.DataFrame(model_rows).sort_values("model")

    metrics_summary_path = Path(cfg["output"]["metrics_summary"])
    fold_metrics_path = Path(cfg["output"]["fold_metrics"])
    raw_summary_path = Path(cfg["output"].get("raw_metrics", metrics_summary_path.parent / "raw_metrics.csv"))
    ensure_dir(metrics_summary_path.parent)
    ensure_dir(fold_metrics_path.parent)
    model_summary_df.to_csv(metrics_summary_path, index=False)
    raw_summary_df.to_csv(raw_summary_path, index=False)
    fold_df.to_csv(fold_metrics_path, index=False)
    print(
        {
            "scanned": scanned,
            "included": len(raw_summary_df),
            "models": len(model_summary_df),
            "filters": {k: v for k, v in cfg.get("input", {}).items() if k != "runs_dir"},
            "metrics_summary": str(metrics_summary_path),
            "fold_metrics": str(fold_metrics_path),
            "raw_metrics": str(raw_summary_path),
        },
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SCG-BP run metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
