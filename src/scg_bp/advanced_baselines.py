from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import (
    TARGET_COLUMNS,
    _features,
    _fit_subject_mean,
    _fold_frames,
    _load_split,
    _mae,
    _predict_subject_mean,
    _write_predictions,
    _write_subject_errors,
)
from .config import load_with_overrides
from .utils import ensure_dir, now_tag, save_json, set_seed


def _build_models(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("advanced_baselines requires scikit-learn. Install scikit-learn>=1.3.") from e

    model_cfg = cfg["models"]
    seed = int(model_cfg.get("random_seed", 42))
    n_estimators = int(model_cfg.get("n_estimators", 500))
    min_samples_leaf = int(model_cfg.get("min_samples_leaf", 2))
    max_features = model_cfg.get("max_features", "sqrt")
    hist_max_iter = int(model_cfg.get("hist_max_iter", 300))
    hist_lr = float(model_cfg.get("hist_learning_rate", 0.03))
    hist_l2 = float(model_cfg.get("hist_l2_regularization", 0.1))
    return {
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
        ),
        "hist_gbr": lambda: MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=hist_max_iter,
                learning_rate=hist_lr,
                l2_regularization=hist_l2,
                random_state=seed,
            )
        ),
    }


def _write_fold_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fit_predict(
    model_factory: Any,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    residual: bool,
) -> np.ndarray:
    model = model_factory()
    if not residual:
        model.fit(x_train, y_train)
        return np.asarray(model.predict(x_eval), dtype=np.float32)
    global_mean, subject_means = _fit_subject_mean(train_df)
    base_train = _predict_subject_mean(train_df, global_mean, subject_means)
    base_eval = _predict_subject_mean(eval_df, global_mean, subject_means)
    model.fit(x_train, y_train - base_train)
    return np.asarray(base_eval + model.predict(x_eval), dtype=np.float32)


def _run_one(
    run_dir: Path,
    model_name: str,
    model_factory: Any,
    residual: bool,
    trainval: pd.DataFrame,
    folds: pd.DataFrame,
    test: pd.DataFrame,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    runtime_seconds: float,
) -> dict[str, Any]:
    trainval_index = {sample_id: i for i, sample_id in enumerate(trainval["sample_id"].astype(str))}
    fold_rows: list[dict[str, Any]] = []
    for fold in sorted(int(x) for x in folds["fold"].unique()):
        train_df, val_df = _fold_frames(trainval, folds, fold)
        train_idx = [trainval_index[x] for x in train_df["sample_id"].astype(str)]
        val_idx = [trainval_index[x] for x in val_df["sample_id"].astype(str)]
        pred = _fit_predict(
            model_factory,
            train_df,
            val_df,
            x_trainval[train_idx],
            y_trainval[train_idx],
            x_trainval[val_idx],
            residual,
        )
        mae_sbp, mae_dbp = _mae(pred, y_trainval[val_idx])
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold,
                "val_mae_sbp": mae_sbp,
                "val_mae_dbp": mae_dbp,
                "val_loss": float((mae_sbp + mae_dbp) / 2),
                "run_dir": str(run_dir),
            }
        )
    test_pred = _fit_predict(model_factory, trainval, test, x_trainval, y_trainval, x_test, residual)
    test_sbp, test_dbp = _mae(test_pred, y_test)
    _write_fold_metrics(run_dir / "fold_metrics.csv", fold_rows)
    _write_predictions(run_dir / "test_predictions.csv", test, test_pred)
    pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    _write_subject_errors(run_dir / "test_subject_errors.csv", pred_df)
    metrics = {
        "model": model_name,
        "mode": "all",
        "fold": None,
        "device": "sklearn",
        "run_dir": str(run_dir),
        "folds": len(fold_rows),
        "cv_val_mae_sbp": float(np.mean([r["val_mae_sbp"] for r in fold_rows])),
        "cv_val_mae_dbp": float(np.mean([r["val_mae_dbp"] for r in fold_rows])),
        "test_mae_sbp": test_sbp,
        "test_mae_dbp": test_dbp,
        "test_loss": float((test_sbp + test_dbp) / 2),
        "runtime_seconds": runtime_seconds,
        "residual_target": residual,
    }
    save_json(run_dir / "metrics.json", metrics)
    return metrics


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    set_seed(int(cfg.get("models", {}).get("random_seed", 42)))
    split_dir = Path(cfg["input"]["split_dir"])
    trainval, folds, test = _load_split(split_dir)
    runs_dir = ensure_dir(cfg["output"]["runs_dir"])
    print(
        {
            "start": "advanced_baselines",
            "split_dir": str(split_dir),
            "trainval": len(trainval),
            "test": len(test),
            "folds": int(folds["fold"].nunique()),
            "runs_dir": str(runs_dir),
        },
        flush=True,
    )
    t0 = time.time()
    array_cache: dict[str, np.ndarray] = {}
    x_trainval, y_trainval = _features(trainval, cfg, array_cache)
    x_test, y_test = _features(test, cfg, array_cache)
    feature_seconds = time.time() - t0
    model_factories = _build_models(cfg)
    tag = now_tag()
    summary_rows: list[dict[str, Any]] = []
    for base_name, factory in model_factories.items():
        for residual in (False, True):
            model_name = f"{base_name}{'_residual' if residual else ''}"
            run_dir = ensure_dir(runs_dir / f"{tag}_{model_name}")
            save_json(run_dir / "config.resolved.json", cfg)
            metrics = _run_one(
                run_dir,
                model_name,
                factory,
                residual,
                trainval,
                folds,
                test,
                x_trainval,
                y_trainval,
                x_test,
                y_test,
                time.time() - t0,
            )
            row = {
                "model": model_name,
                "cv_val_mae_sbp": metrics["cv_val_mae_sbp"],
                "cv_val_mae_dbp": metrics["cv_val_mae_dbp"],
                "cv_val_mae_mean": (metrics["cv_val_mae_sbp"] + metrics["cv_val_mae_dbp"]) / 2,
                "test_mae_sbp": metrics["test_mae_sbp"],
                "test_mae_dbp": metrics["test_mae_dbp"],
                "test_mae_mean": metrics["test_loss"],
                "run_dir": str(run_dir),
            }
            summary_rows.append(row)
            print(row, flush=True)
    summary = pd.DataFrame(summary_rows).sort_values("test_mae_mean")
    summary.to_csv(runs_dir / f"{tag}_advanced_baseline_summary.csv", index=False)
    print({"done": "advanced_baselines", "feature_seconds": feature_seconds, "summary_rows": len(summary_rows)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stronger sklearn feature/residual baselines for SCG-BP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
