from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_with_overrides
from .train import _standardize_window
from .utils import ensure_dir, now_tag, save_json, set_seed


TARGET_COLUMNS = ["SBP", "DBP"]


def _load_split(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trainval = pd.read_csv(split_dir / "trainval.csv")
    folds = pd.read_csv(split_dir / "folds.csv")
    test = pd.read_csv(split_dir / "test.csv")
    return trainval, folds, test


def _fold_frames(trainval: pd.DataFrame, folds: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = folds[folds["fold"] == fold][["sample_id", "subset"]]
    merged = trainval.merge(fold_map, on="sample_id", how="inner")
    train_df = merged[merged["subset"] == "train"].drop(columns=["subset"])
    val_df = merged[merged["subset"] == "val"].drop(columns=["subset"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _window_array(row: pd.Series, window_size: int, input_channels: int, array_cache: dict[str, np.ndarray]) -> np.ndarray:
    array_path = str(row["signal_array"])
    if array_path not in array_cache:
        array_cache[array_path] = np.load(array_path, mmap_mode="r")
    full = array_cache[array_path]
    start = max(0, min(int(row["start_row"]), max(0, full.shape[0] - 1)))
    end = max(start + 1, min(int(row["end_row"]), full.shape[0]))
    arr = np.asarray(full[start:end], dtype=np.float32)
    return _standardize_window(arr, window_size, input_channels)


def _feature_names(input_channels: int, include_hr: bool, include_window_meta: bool) -> list[str]:
    stats = ["mean", "std", "min", "max", "median", "q25", "q75", "rms", "abs_mean", "diff_mean", "diff_std"]
    names = [f"ch{ch}_{name}" for name in stats for ch in range(input_channels)]
    if include_hr:
        names.append("HR")
    if include_window_meta:
        names.extend(["window_offset_sec", "window_size"])
    return names


def _extract_one(row: pd.Series, window_size: int, input_channels: int, include_hr: bool, include_window_meta: bool, array_cache: dict[str, np.ndarray]) -> np.ndarray:
    arr = _window_array(row, window_size, input_channels, array_cache)
    chunks: list[np.ndarray] = [
        arr.mean(axis=0),
        arr.std(axis=0),
        arr.min(axis=0),
        arr.max(axis=0),
        np.median(arr, axis=0),
        np.quantile(arr, 0.25, axis=0),
        np.quantile(arr, 0.75, axis=0),
        np.sqrt(np.mean(arr * arr, axis=0)),
        np.mean(np.abs(arr), axis=0),
    ]
    diff = np.diff(arr, axis=0)
    if len(diff) == 0:
        diff = np.zeros((1, input_channels), dtype=np.float32)
    chunks.extend([diff.mean(axis=0), diff.std(axis=0)])
    values = np.concatenate(chunks).astype(np.float32)
    extras: list[float] = []
    if include_hr:
        extras.append(float(row.get("HR", 0.0) or 0.0))
    if include_window_meta:
        extras.append(float(row.get("window_offset_sec", 0.0) or 0.0))
        extras.append(float(row.get("window_size", window_size) or window_size))
    if extras:
        values = np.concatenate([values, np.asarray(extras, dtype=np.float32)])
    return values


def _features(df: pd.DataFrame, cfg: dict[str, Any], array_cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feature_cfg = cfg["features"]
    input_channels = int(feature_cfg["input_channels"])
    window_size = int(feature_cfg["window_size"])
    include_hr = bool(feature_cfg.get("include_hr", False))
    include_window_meta = bool(feature_cfg.get("include_window_meta", True))
    x = np.vstack(
        [
            _extract_one(row, window_size, input_channels, include_hr, include_window_meta, array_cache)
            for _, row in df.iterrows()
        ]
    ).astype(np.float32)
    y = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    return x, y


def _mae(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    err = np.abs(pred - y)
    return float(err[:, 0].mean()), float(err[:, 1].mean())


def _fit_mean(y_train: np.ndarray) -> np.ndarray:
    return y_train.mean(axis=0, keepdims=True)


def _predict_mean(mean: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(mean, n, axis=0)


def _standardize_x(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_eval - mean) / std, mean, std


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((len(x_train), 1), dtype=np.float32), x_train], axis=1)
    penalty = np.eye(x_aug.shape[1], dtype=np.float32) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + penalty, x_aug.T @ y_train)


def _predict_ridge(x_eval: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((len(x_eval), 1), dtype=np.float32), x_eval], axis=1)
    return x_aug @ coef


def _ridge_eval(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float) -> np.ndarray:
    x_tr, x_ev, _, _ = _standardize_x(x_train, x_eval)
    coef = _fit_ridge(x_tr, y_train, alpha)
    return _predict_ridge(x_ev, coef)


def _select_alpha(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, alphas: list[float]) -> tuple[float, np.ndarray, float]:
    best_alpha = float(alphas[0])
    best_pred = _ridge_eval(x_train, y_train, x_val, best_alpha)
    best_score = sum(_mae(best_pred, y_val))
    for alpha in alphas[1:]:
        pred = _ridge_eval(x_train, y_train, x_val, float(alpha))
        score = sum(_mae(pred, y_val))
        if score < best_score:
            best_alpha = float(alpha)
            best_pred = pred
            best_score = score
    return best_alpha, best_pred, float(best_score)


def _write_fold_metrics(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with (run_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path: Path, df: pd.DataFrame, pred: np.ndarray) -> None:
    out = df[["sample_id", "subject_id", "SBP", "DBP"]].copy()
    out["pred_SBP"] = pred[:, 0]
    out["pred_DBP"] = pred[:, 1]
    out["abs_err_SBP"] = np.abs(out["pred_SBP"] - out["SBP"].astype(float))
    out["abs_err_DBP"] = np.abs(out["pred_DBP"] - out["DBP"].astype(float))
    out.to_csv(path, index=False)


def _write_subject_errors(path: Path, pred_df: pd.DataFrame) -> None:
    rows = []
    for subject, g in pred_df.groupby("subject_id"):
        rows.append(
            {
                "subject_id": subject,
                "n": int(len(g)),
                "mae_sbp": float(g["abs_err_SBP"].mean()),
                "mae_dbp": float(g["abs_err_DBP"].mean()),
                "true_sbp_mean": float(g["SBP"].mean()),
                "true_dbp_mean": float(g["DBP"].mean()),
                "pred_sbp_mean": float(g["pred_SBP"].mean()),
                "pred_dbp_mean": float(g["pred_DBP"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_mean_baseline(
    run_dir: Path,
    trainval: pd.DataFrame,
    folds: pd.DataFrame,
    test: pd.DataFrame,
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    runtime_seconds: float,
) -> None:
    fold_rows = []
    for fold in sorted(int(x) for x in folds["fold"].unique()):
        train_df, val_df = _fold_frames(trainval, folds, fold)
        y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        y_val = val_df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        pred = _predict_mean(_fit_mean(y_train), len(y_val))
        mae_sbp, mae_dbp = _mae(pred, y_val)
        fold_rows.append({"model": "mean_baseline", "fold": fold, "val_mae_sbp": mae_sbp, "val_mae_dbp": mae_dbp, "val_loss": float((mae_sbp + mae_dbp) / 2), "run_dir": str(run_dir)})
    train_y = trainval[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    test_y = test[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    test_pred = _predict_mean(_fit_mean(train_y), len(test_y))
    test_sbp, test_dbp = _mae(test_pred, test_y)
    _write_fold_metrics(run_dir, fold_rows)
    _write_predictions(run_dir / "test_predictions.csv", test, test_pred)
    pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    _write_subject_errors(run_dir / "test_subject_errors.csv", pred_df)
    save_json(
        run_dir / "metrics.json",
        {
            "model": "mean_baseline",
            "mode": "all",
            "fold": None,
            "device": "numpy",
            "run_dir": str(run_dir),
            "folds": len(fold_rows),
            "cv_val_mae_sbp": float(np.mean([r["val_mae_sbp"] for r in fold_rows])),
            "cv_val_mae_dbp": float(np.mean([r["val_mae_dbp"] for r in fold_rows])),
            "test_mae_sbp": test_sbp,
            "test_mae_dbp": test_dbp,
            "test_loss": float((test_sbp + test_dbp) / 2),
            "runtime_seconds": runtime_seconds,
        },
    )


def _run_ridge_baseline(
    run_dir: Path,
    trainval: pd.DataFrame,
    folds: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict[str, Any],
    features: dict[str, tuple[np.ndarray, np.ndarray]],
    runtime_seconds: float,
) -> None:
    alphas = [float(x) for x in cfg["ridge"]["alphas"]]
    feature_names = _feature_names(
        int(cfg["features"]["input_channels"]),
        bool(cfg["features"].get("include_hr", False)),
        bool(cfg["features"].get("include_window_meta", True)),
    )
    x_trainval, y_trainval = features["trainval"]
    x_test, y_test = features["test"]
    trainval_index = {sample_id: i for i, sample_id in enumerate(trainval["sample_id"].astype(str))}
    fold_rows = []
    for fold in sorted(int(x) for x in folds["fold"].unique()):
        train_df, val_df = _fold_frames(trainval, folds, fold)
        train_idx = [trainval_index[x] for x in train_df["sample_id"].astype(str)]
        val_idx = [trainval_index[x] for x in val_df["sample_id"].astype(str)]
        alpha, pred, _ = _select_alpha(x_trainval[train_idx], y_trainval[train_idx], x_trainval[val_idx], y_trainval[val_idx], alphas)
        mae_sbp, mae_dbp = _mae(pred, y_trainval[val_idx])
        fold_rows.append({"model": "ridge_features", "fold": fold, "val_mae_sbp": mae_sbp, "val_mae_dbp": mae_dbp, "val_loss": float((mae_sbp + mae_dbp) / 2), "alpha": alpha, "run_dir": str(run_dir)})
    final_fold = int(cfg["ridge"].get("final_validation_fold", 1))
    final_train, final_val = _fold_frames(trainval, folds, final_fold)
    final_train_idx = [trainval_index[x] for x in final_train["sample_id"].astype(str)]
    final_val_idx = [trainval_index[x] for x in final_val["sample_id"].astype(str)]
    final_alpha, _, _ = _select_alpha(x_trainval[final_train_idx], y_trainval[final_train_idx], x_trainval[final_val_idx], y_trainval[final_val_idx], alphas)
    test_pred = _ridge_eval(x_trainval, y_trainval, x_test, final_alpha)
    test_sbp, test_dbp = _mae(test_pred, y_test)
    _write_fold_metrics(run_dir, fold_rows)
    _write_predictions(run_dir / "test_predictions.csv", test, test_pred)
    pred_df = pd.read_csv(run_dir / "test_predictions.csv")
    _write_subject_errors(run_dir / "test_subject_errors.csv", pred_df)
    save_json(run_dir / "feature_names.json", {"feature_names": feature_names})
    save_json(
        run_dir / "metrics.json",
        {
            "model": "ridge_features",
            "mode": "all",
            "fold": None,
            "device": "numpy",
            "run_dir": str(run_dir),
            "folds": len(fold_rows),
            "cv_val_mae_sbp": float(np.mean([r["val_mae_sbp"] for r in fold_rows])),
            "cv_val_mae_dbp": float(np.mean([r["val_mae_dbp"] for r in fold_rows])),
            "test_mae_sbp": test_sbp,
            "test_mae_dbp": test_dbp,
            "test_loss": float((test_sbp + test_dbp) / 2),
            "runtime_seconds": runtime_seconds,
            "alphas": alphas,
            "final_alpha": final_alpha,
            "final_validation_fold": final_fold,
            "feature_count": len(feature_names),
        },
    )


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    set_seed(42)
    split_dir = Path(cfg["input"]["split_dir"])
    trainval, folds, test = _load_split(split_dir)
    tag = now_tag()
    runs_dir = ensure_dir(cfg["output"]["runs_dir"])
    print(
        {
            "start": "baselines",
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
    features = {
        "trainval": _features(trainval, cfg, array_cache),
        "test": _features(test, cfg, array_cache),
    }
    feature_seconds = time.time() - t0
    mean_dir = ensure_dir(runs_dir / f"{tag}_mean_baseline")
    ridge_dir = ensure_dir(runs_dir / f"{tag}_ridge_features")
    save_json(mean_dir / "config.resolved.json", cfg)
    save_json(ridge_dir / "config.resolved.json", cfg)
    _run_mean_baseline(mean_dir, trainval, folds, test, features, feature_seconds)
    _run_ridge_baseline(ridge_dir, trainval, folds, test, cfg, features, time.time() - t0)
    print({"done": "baselines", "mean_run": str(mean_dir), "ridge_run": str(ridge_dir)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-neural SCG-BP baselines.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
