from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir


def plot_mae_bars(df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = df.sort_values("model")
    x = range(len(plot_df))

    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.15 for i in x], plot_df["test_mae_sbp"], width=0.3, label="SBP MAE")
    plt.bar([i + 0.15 for i in x], plot_df["test_mae_dbp"], width=0.3, label="DBP MAE")
    plt.xticks(list(x), plot_df["model"].tolist(), rotation=20)
    plt.ylabel("MAE")
    plt.title("Model Comparison on Holdout Test")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "test_mae_comparison.png", dpi=160)
    plt.close()


def plot_cv_scatter(fold_df: pd.DataFrame, figure_dir: Path) -> None:
    if fold_df.empty:
        return
    plt.figure(figsize=(8, 5))
    for model, g in fold_df.groupby("model"):
        plt.scatter(g["val_mae_sbp"], g["val_mae_dbp"], label=model)
    plt.xlabel("Fold Val MAE SBP")
    plt.ylabel("Fold Val MAE DBP")
    plt.title("CV Fold Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "cv_fold_scatter.png", dpi=160)
    plt.close()


def write_report(summary_df: pd.DataFrame, report_path: Path) -> None:
    lines = []
    lines.append("# SCG Blood Pressure Prediction Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("- Models: full, cnn_only, lstm_only, mlp_only")
    lines.append("- Metrics: SBP/DBP MAE on holdout test")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")

    keep_cols = ["model", "cv_val_mae_sbp", "cv_val_mae_dbp", "test_mae_sbp", "test_mae_dbp", "run_dir"]
    table = summary_df[keep_cols].copy()
    lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("## Literature Reference (Not Reproduced)")
    lines.append("- Biosensors 2024: Continuous Estimation of Blood Pressure by Utilizing Seismocardiogram Signal Features in Relation to Electrocardiogram.")
    lines.append("- Annual Review of Biomedical Engineering 2022: Cuffless Blood Pressure Measurement.")
    lines.append("- Note: These are cited references, not reproduced under the same split/protocol in this run.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)

    summary_path = Path(cfg["input"]["metrics_summary"])
    fold_path = Path(cfg["input"]["fold_metrics"])
    summary_df = pd.read_csv(summary_path)
    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()

    fig_dir = ensure_dir(cfg["output"]["figure_dir"])
    report_path = Path(cfg["output"]["report_md"])
    ensure_dir(report_path.parent)

    plot_mae_bars(summary_df, fig_dir)
    plot_cv_scatter(fold_df, fig_dir)
    write_report(summary_df, report_path)

    print({"figures": str(fig_dir), "report": str(report_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures and markdown report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
