from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir


def _col(df: pd.DataFrame, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def plot_mae_bars(df: pd.DataFrame, figure_dir: Path) -> None:
    plot_df = df.sort_values("model")
    sbp_col = _col(plot_df, "test_mae_sbp", "cv_val_mae_sbp_mean")
    dbp_col = _col(plot_df, "test_mae_dbp", "cv_val_mae_dbp_mean")
    x = range(len(plot_df))
    plt.figure(figsize=(10, 5))
    plt.bar([i - 0.15 for i in x], plot_df[sbp_col], width=0.3, label="SBP MAE")
    plt.bar([i + 0.15 for i in x], plot_df[dbp_col], width=0.3, label="DBP MAE")
    plt.xticks(list(x), plot_df["model"].tolist(), rotation=20)
    plt.ylabel("MAE")
    plt.title("SCG-BP Ablation Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "mae_barplot.png", dpi=160)
    plt.close()


def plot_cv_scatter(fold_df: pd.DataFrame, figure_dir: Path) -> None:
    if fold_df.empty or "val_mae_sbp" not in fold_df.columns or "val_mae_dbp" not in fold_df.columns:
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
    lines = [
        "# SCG Blood Pressure Prediction Report",
        "",
        "## Summary",
        "- Models: full, cnn_only, lstm_only, mlp_only",
        "- Primary metrics: SBP MAE, DBP MAE, mean MAE",
        "- Data caveat: current labels are limited; results are suitable for project comparison, not strong clinical generalization claims.",
        "",
        "## Ablation Results",
        "",
    ]
    keep_cols = [
        c
        for c in [
            "model",
            "cv_folds",
            "cv_val_mae_sbp_mean",
            "cv_val_mae_dbp_mean",
            "cv_val_mae_mean",
            "test_mae_sbp",
            "test_mae_dbp",
            "test_mae_mean",
            "runtime_seconds_total",
        ]
        if c in summary_df.columns
    ]
    table = summary_df[keep_cols].copy()
    lines.append(table.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Protocol Notes",
            "- Subject-level holdout and GroupKFold are used to avoid subject leakage.",
            "- SCG windows are generated from preprocessed arrays, not raw CSV during training.",
            "- Alignment method is recorded in the processed window index; default is rank_interpolation when exact timestamp alignment is unavailable.",
            "",
            "## Literature Reference (Not Reproduced)",
            "- Biosensors 2024: Continuous Estimation of Blood Pressure by Utilizing Seismocardiogram Signal Features in Relation to Electrocardiogram.",
            "- Annual Review of Biomedical Engineering 2022: Cuffless Blood Pressure Measurement.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    summary_path = Path(cfg["input"]["metrics_summary"])
    fold_path = Path(cfg["input"]["fold_metrics"])
    summary_df = pd.read_csv(summary_path)
    fold_df = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
    fig_dir = ensure_dir(cfg["output"]["figure_dir"])
    report_path = Path(cfg["output"]["report_md"])
    xlsx_path = Path(cfg["output"].get("summary_xlsx", report_path.parent / "ablation_summary.xlsx"))
    csv_path = Path(cfg["output"].get("summary_csv", report_path.parent / "ablation_summary.csv"))
    ensure_dir(report_path.parent)
    plot_mae_bars(summary_df, fig_dir)
    plot_cv_scatter(fold_df, fig_dir)
    write_report(summary_df, report_path)
    summary_df.to_csv(csv_path, index=False)
    try:
        summary_df.to_excel(xlsx_path, index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Skip xlsx export: {type(e).__name__}: {e}", flush=True)
    print({"figures": str(fig_dir), "report": str(report_path), "summary_csv": str(csv_path), "summary_xlsx": str(xlsx_path)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SCG-BP ablation report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
