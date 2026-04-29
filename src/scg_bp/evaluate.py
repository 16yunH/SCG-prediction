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
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["metrics_path"] = str(path)
    data["run_dir"] = data.get("run_dir", str(path.parent))
    return data


def _mean_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _std_numeric(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.std(ddof=0)) if len(vals) > 1 else float("nan")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    runs_dir = Path(cfg["input"]["runs_dir"])
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    summaries: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        summaries.append(_load_metrics(metrics_path))
        fold_path = metrics_path.parent / "fold_metrics.csv"
        if fold_path.exists():
            df = pd.read_csv(fold_path)
            df["metrics_path"] = str(metrics_path)
            df["run_dir"] = str(metrics_path.parent)
            fold_frames.append(df)

    if not summaries:
        raise RuntimeError(f"No metrics.json found under {runs_dir}")

    raw_summary_df = pd.DataFrame(summaries)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    model_rows = []
    for model, g in raw_summary_df.groupby("model", dropna=False):
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
            "runtime_seconds_total": _mean_numeric(g.get("runtime_seconds", pd.Series(dtype=float))) * len(g) if "runtime_seconds" in g else float("nan"),
        }
        row["cv_val_mae_mean"] = float(np.nanmean([row["cv_val_mae_sbp_mean"], row["cv_val_mae_dbp_mean"]]))
        row["test_mae_mean"] = float(np.nanmean([row["test_mae_sbp"], row["test_mae_dbp"]]))
        model_rows.append(row)
    model_summary_df = pd.DataFrame(model_rows).sort_values("model")

    metrics_summary_path = Path(cfg["output"]["metrics_summary"])
    fold_metrics_path = Path(cfg["output"]["fold_metrics"])
    raw_summary_path = Path(cfg["output"].get("raw_metrics", metrics_summary_path.parent / "raw_metrics.csv"))
    ensure_dir(metrics_summary_path.parent)
    ensure_dir(fold_metrics_path.parent)
    model_summary_df.to_csv(metrics_summary_path, index=False)
    raw_summary_df.to_csv(raw_summary_path, index=False)
    fold_df.to_csv(fold_metrics_path, index=False)
    print({"runs": len(raw_summary_df), "models": len(model_summary_df), "metrics_summary": str(metrics_summary_path), "fold_metrics": str(fold_metrics_path), "raw_metrics": str(raw_summary_path)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate SCG-BP run metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
