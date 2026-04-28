from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)

    runs_dir = Path(cfg["input"]["runs_dir"])
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir not found: {runs_dir}")

    summaries = []
    fold_frames = []

    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        metrics_path = run_dir / "metrics.json"
        fold_path = run_dir / "fold_metrics.csv"

        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                m = json.load(f)
            summaries.append(m)

        if fold_path.exists():
            df = pd.read_csv(fold_path)
            df["run_dir"] = str(run_dir)
            fold_frames.append(df)

    if not summaries:
        raise RuntimeError(f"No metrics.json found under {runs_dir}")

    summary_df = pd.DataFrame(summaries)
    fold_df = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    metrics_summary_path = Path(cfg["output"]["metrics_summary"])
    fold_metrics_path = Path(cfg["output"]["fold_metrics"])
    ensure_dir(metrics_summary_path.parent)
    ensure_dir(fold_metrics_path.parent)

    summary_df.to_csv(metrics_summary_path, index=False)
    fold_df.to_csv(fold_metrics_path, index=False)

    print(
        {
            "runs": len(summary_df),
            "metrics_summary": str(metrics_summary_path),
            "fold_metrics": str(fold_metrics_path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
