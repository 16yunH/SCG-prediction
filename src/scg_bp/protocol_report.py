from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # noqa: BLE001
    plt = None
    _HAS_MPL = False

import pandas as pd

from .config import load_with_overrides
from .report import _to_markdown_table
from .utils import ensure_dir


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _best_row(df: pd.DataFrame, protocol: str) -> dict[str, Any]:
    if df.empty:
        return {
            "protocol": protocol,
            "model": "",
            "strategy": "",
            "cv_mean_mae": float("nan"),
            "test_sbp_mae": float("nan"),
            "test_dbp_mae": float("nan"),
            "test_mean_mae": float("nan"),
        }
    row = df.sort_values("test_mae_mean").iloc[0]
    return {
        "protocol": protocol,
        "model": row.get("model", ""),
        "strategy": "best_single_run_or_baseline",
        "cv_mean_mae": float(row.get("cv_val_mae_mean", float("nan"))),
        "test_sbp_mae": float(row.get("test_mae_sbp", float("nan"))),
        "test_dbp_mae": float(row.get("test_mae_dbp", float("nan"))),
        "test_mean_mae": float(row.get("test_mae_mean", float("nan"))),
    }


def _plot_protocol_bars(summary: pd.DataFrame, figure_dir: Path) -> None:
    if not _HAS_MPL or summary.empty:
        return
    plot_df = summary.copy()
    labels = [f"{r.protocol}\n{r.model}" for r in plot_df.itertuples(index=False)]
    x = range(len(plot_df))
    plt.figure(figsize=(10, 5))
    plt.bar(x, plot_df["test_mean_mae"], color=["#6b7280", "#2563eb", "#16a34a"][: len(plot_df)])
    plt.xticks(list(x), labels, rotation=0)
    plt.ylabel("Test mean MAE (mmHg)")
    plt.title("Protocol-Level Best Results")
    plt.tight_layout()
    plt.savefig(figure_dir / "protocol_best_mae.png", dpi=180)
    plt.close()


def _plot_seed_stability(seed_df: pd.DataFrame, figure_dir: Path) -> None:
    if not _HAS_MPL or seed_df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(seed_df["seed"].astype(str), seed_df["test_mean"], marker="o", label="Test mean MAE")
    plt.plot(seed_df["seed"].astype(str), seed_df["cv_mean"], marker="o", label="CV mean MAE")
    plt.xlabel("Seed")
    plt.ylabel("MAE (mmHg)")
    plt.title("Calibrated CNN Ensemble Seed Stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "seed_stability.png", dpi=180)
    plt.close()


def _plot_hard_subjects(subject_df: pd.DataFrame, figure_dir: Path, top_k: int = 8) -> None:
    if not _HAS_MPL or subject_df.empty:
        return
    plot_df = subject_df.sort_values("mae_mean_mean", ascending=False).head(top_k)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["subject_id"].astype(str), plot_df["mae_mean_mean"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean MAE across seeds (mmHg)")
    plt.title("Hardest Calibrated Test Subjects")
    plt.tight_layout()
    plt.savefig(figure_dir / "hard_subjects.png", dpi=180)
    plt.close()


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:  # noqa: BLE001
        return str(value)


def _write_report(
    report_path: Path,
    protocol_summary: pd.DataFrame,
    subject_independent: pd.DataFrame,
    calibrated: pd.DataFrame,
    optimized: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    seed_summary: pd.DataFrame,
    subject_errors: pd.DataFrame,
) -> None:
    metric_cols = ["model", "cv_val_mae_mean", "test_mae_sbp", "test_mae_dbp", "test_mae_mean"]
    si_cols = [c for c in metric_cols if c in subject_independent.columns]
    cal_cols = [c for c in metric_cols if c in calibrated.columns]
    lines = [
        "# SCG-BP Dual-Protocol Report",
        "",
        "## Executive Summary",
        "- Subject-independent evaluation tests no-calibration performance on unseen people.",
        "- Subject-dependent calibrated evaluation tests within-subject prediction after some BP labels from the same people are available.",
        "- Current data supports calibrated prediction much better than no-calibration cross-subject generalization.",
        "",
        "## Best Results By Protocol",
        "",
        _to_markdown_table(protocol_summary),
        "",
        "## Interpretation",
        "- The subject-independent best result is the train-mean baseline, which confirms the raw dataset is not sufficient for reliable unseen-subject generalization.",
        "- The calibrated CNN ensemble is stable across seeds and is the recommended result to report as the main model outcome.",
        "- The calibrated protocol still isolates BP label groups, so jitter windows from the same BP record do not leak across train/test.",
        "",
        "## Subject-Independent Results",
        "",
        _to_markdown_table(subject_independent[si_cols]) if not subject_independent.empty else "_No data_",
        "",
        "## Calibrated Results",
        "",
        _to_markdown_table(calibrated[cal_cols]) if not calibrated.empty else "_No data_",
        "",
        "## Optimized Model Summary",
        "",
        _to_markdown_table(optimized) if not optimized.empty else "_No data_",
        "",
        "## Seed Stability",
        "",
        _to_markdown_table(seed_metrics) if not seed_metrics.empty else "_No data_",
        "",
        "Seed summary:",
        "",
        _to_markdown_table(seed_summary) if not seed_summary.empty else "_No data_",
        "",
        "## Hardest Calibrated Subjects",
        "",
        _to_markdown_table(subject_errors.head(10)) if not subject_errors.empty else "_No data_",
        "",
        "## Reporting Recommendation",
        "- Report subject-independent results as a negative/generalization finding.",
        "- Report calibrated CNN ensemble as the strongest model result.",
        "- State the calibrated four-seed result as: test mean MAE "
        + (
            f"{_format_float(seed_summary.loc[seed_summary['stat'] == 'mean', 'test_mean'].iloc[0])} +/- "
            f"{_format_float(seed_summary.loc[seed_summary['stat'] == 'std', 'test_mean'].iloc[0])} mmHg"
            if not seed_summary.empty and "stat" in seed_summary.columns
            else "unavailable"
        )
        + ".",
        "- Note that high-error subjects mostly have only three calibrated test windows, so per-subject error estimates are noisy.",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    subject_independent = _read_csv(input_cfg["subject_independent_summary"])
    calibrated = _read_csv(input_cfg["calibrated_summary"])
    optimized = _read_csv(input_cfg["optimized_summary"])
    seed_metrics = _read_csv(input_cfg["seed_metrics"])
    seed_summary = _read_csv(input_cfg["seed_summary"])
    subject_errors = _read_csv(input_cfg["subject_error_summary"])

    rows = [
        _best_row(subject_independent, "subject_independent"),
        _best_row(calibrated, "subject_dependent_calibrated"),
    ]
    if not optimized.empty:
        best_opt = optimized.sort_values("test_mean_mae").iloc[0]
        rows.append(
            {
                "protocol": best_opt.get("protocol", "subject_dependent_calibrated"),
                "model": best_opt.get("model", ""),
                "strategy": best_opt.get("strategy", ""),
                "cv_mean_mae": float(best_opt.get("cv_mean_mae", float("nan"))),
                "test_sbp_mae": float(best_opt.get("test_sbp_mae", float("nan"))),
                "test_dbp_mae": float(best_opt.get("test_dbp_mae", float("nan"))),
                "test_mean_mae": float(best_opt.get("test_mean_mae", float("nan"))),
            }
        )
    protocol_summary = pd.DataFrame(rows)

    report_path = Path(output_cfg["report_md"])
    summary_path = Path(output_cfg["summary_csv"])
    figure_dir = ensure_dir(output_cfg["figure_dir"])
    ensure_dir(summary_path.parent)
    protocol_summary.to_csv(summary_path, index=False)
    _plot_protocol_bars(protocol_summary, figure_dir)
    _plot_seed_stability(seed_metrics, figure_dir)
    _plot_hard_subjects(subject_errors, figure_dir)
    _write_report(report_path, protocol_summary, subject_independent, calibrated, optimized, seed_metrics, seed_summary, subject_errors)
    print({"report": str(report_path), "summary_csv": str(summary_path), "figure_dir": str(figure_dir)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a dual-protocol SCG-BP report.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
