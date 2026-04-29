from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import load_with_overrides
from .io_scg import read_scg_full_array, read_scg_window
from .models import ModelConfig, build_model
from .utils import ensure_dir, now_tag, save_json, set_seed


def normalize_gpu_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return []


def _standardize_window(arr: np.ndarray, window_size: int, input_channels: int) -> np.ndarray:
    if arr.ndim != 2:
        arr = np.asarray(arr).reshape(-1, input_channels)
    if arr.shape[1] < input_channels:
        pad = np.zeros((arr.shape[0], input_channels - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    arr = arr[:, :input_channels].astype(np.float32, copy=False)
    if arr.shape[0] < window_size:
        pad = np.zeros((window_size - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:window_size, :]
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True) + 1e-6
    return ((arr - mean) / std).astype(np.float32, copy=False)


def _target_stats(df: pd.DataFrame, enabled: bool) -> dict[str, list[float]]:
    if not enabled:
        return {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
    y = df[["SBP", "DBP"]].to_numpy(dtype=np.float32)
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": [float(mean[0]), float(mean[1])], "std": [float(std[0]), float(std[1])]}


def _target_tensors(stats: dict[str, list[float]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 2)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 2)
    return mean, std


class ScgDataset(Dataset):
    def __init__(
        self,
        sample_df: pd.DataFrame,
        input_channels: int,
        window_size: int,
        cache_windows: bool = True,
        mmap_arrays: bool = True,
        allow_csv_fallback: bool = False,
        target_stats: dict[str, list[float]] | None = None,
    ) -> None:
        self.df = sample_df.reset_index(drop=True)
        self.input_channels = input_channels
        self.window_size = window_size
        self.cache_windows = cache_windows
        self.mmap_arrays = mmap_arrays
        self.allow_csv_fallback = allow_csv_fallback
        self.target_mean = np.asarray((target_stats or {"mean": [0.0, 0.0]})["mean"], dtype=np.float32)
        self.target_std = np.asarray((target_stats or {"std": [1.0, 1.0]})["std"], dtype=np.float32)
        self.target_std = np.where(self.target_std < 1e-6, 1.0, self.target_std).astype(np.float32)
        self.array_cache: dict[str, np.ndarray] = {}
        self.x_cache: list[torch.Tensor] | None = None
        self.y_cache: list[torch.Tensor] | None = None
        if self.cache_windows:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _load_full_array(self, row: pd.Series) -> np.ndarray:
        array_path = str(row.get("signal_array", "") or "")
        if array_path and Path(array_path).exists():
            key = array_path
            if key not in self.array_cache:
                self.array_cache[key] = np.load(array_path, mmap_mode="r" if self.mmap_arrays else None)
            return self.array_cache[key]

        if not self.allow_csv_fallback:
            raise FileNotFoundError(f"Missing array file: {array_path}")

        scg_file = str(row.get("scg_file", ""))
        mode = str(row.get("scg_mode", row.get("channel_mode", "7col")))
        key = f"csv::{scg_file}::{mode}"
        if key not in self.array_cache:
            self.array_cache[key] = read_scg_full_array(Path(scg_file), mode, self.input_channels)
        return self.array_cache[key]

    def _row_to_xy(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        start_row = int(row["start_row"])
        end_row = int(row["end_row"])
        try:
            full = self._load_full_array(row)
            start_row = max(0, min(start_row, max(0, full.shape[0] - 1)))
            end_row = max(start_row + 1, min(end_row, full.shape[0]))
            arr = np.asarray(full[start_row:end_row], dtype=np.float32)
        except Exception:
            if not self.allow_csv_fallback:
                raise
            path = Path(row["scg_file"])
            mode = str(row.get("scg_mode", "7col"))
            win = read_scg_window(path, start_row, end_row, mode, self.input_channels)
            arr = win.to_numpy(dtype=np.float32)
        arr = _standardize_window(arr, self.window_size, self.input_channels)
        x = torch.from_numpy(arr.T.copy())
        y_raw = np.asarray([float(row["SBP"]), float(row["DBP"])], dtype=np.float32)
        y = torch.from_numpy(((y_raw - self.target_mean) / self.target_std).astype(np.float32))
        return x, y

    def _build_cache(self) -> None:
        self.x_cache = []
        self.y_cache = []
        for idx in range(len(self.df)):
            x, y = self._row_to_xy(idx)
            self.x_cache.append(x)
            self.y_cache.append(y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.x_cache is not None and self.y_cache is not None:
            return self.x_cache[idx], self.y_cache[idx]
        return self._row_to_xy(idx)


def mae_per_target(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    e = torch.abs(pred - target)
    return float(e[:, 0].mean().item()), float(e[:, 1].mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    target_stats: dict[str, list[float]],
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = nn.SmoothL1Loss()
    target_mean, target_std = _target_tensors(target_stats, device)
    losses: list[float] = []
    sbps: list[float] = []
    dbps: list[float] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train_mode):
            pred = model(x)
            loss = loss_fn(pred, y)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        pred_raw = pred.detach() * target_std + target_mean
        y_raw = y * target_std + target_mean
        sbp, dbp = mae_per_target(pred_raw, y_raw)
        sbps.append(sbp)
        dbps.append(dbp)
    return {
        "loss": float(np.mean(losses) if losses else np.nan),
        "mae_sbp": float(np.mean(sbps) if sbps else np.nan),
        "mae_dbp": float(np.mean(dbps) if dbps else np.nan),
    }


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    workers: int,
    channels: int,
    window_size: int,
    pin_memory: bool,
    cache_windows: bool,
    mmap_arrays: bool,
    allow_csv_fallback: bool,
    target_stats: dict[str, list[float]],
) -> tuple[DataLoader, DataLoader]:
    train_ds = ScgDataset(
        train_df,
        channels,
        window_size,
        cache_windows=cache_windows,
        mmap_arrays=mmap_arrays,
        allow_csv_fallback=allow_csv_fallback,
        target_stats=target_stats,
    )
    val_ds = ScgDataset(
        val_df,
        channels,
        window_size,
        cache_windows=cache_windows,
        mmap_arrays=mmap_arrays,
        allow_csv_fallback=allow_csv_fallback,
        target_stats=target_stats,
    )
    loader_kwargs: dict[str, Any] = {"batch_size": batch_size, "num_workers": workers, "pin_memory": pin_memory}
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(train_ds, shuffle=True, **loader_kwargs), DataLoader(val_ds, shuffle=False, **loader_kwargs)


def fit_one_fold(model_name: str, model_cfg: ModelConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    runtime_cfg = cfg["runtime"]
    opt_cfg = cfg["optimization"]
    target_stats = _target_stats(train_df, enabled=bool(opt_cfg.get("target_standardize", True)))
    model = build_model(model_name, model_cfg).to(device)
    gpu_ids = normalize_gpu_ids(runtime_cfg.get("gpu_ids", []))
    if device.type == "cuda" and bool(runtime_cfg.get("use_data_parallel", False)) and len(gpu_ids) > 1:
        visible = torch.cuda.device_count()
        dp_ids = [i for i in gpu_ids if i < visible]
        if len(dp_ids) > 1:
            model = nn.DataParallel(model, device_ids=dp_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]))
    train_loader, val_loader = build_loaders(
        train_df,
        val_df,
        int(opt_cfg["batch_size"]),
        int(runtime_cfg["num_workers"]),
        model_cfg.input_channels,
        model_cfg.window_size,
        bool(runtime_cfg.get("pin_memory", True)),
        bool(runtime_cfg.get("cache_windows", True)),
        bool(runtime_cfg.get("mmap_arrays", True)),
        bool(runtime_cfg.get("allow_csv_fallback", False)),
        target_stats,
    )
    best_val = float("inf")
    best_state = None
    patience = int(opt_cfg["early_stop_patience"])
    bad = 0
    history: list[dict[str, float]] = []
    print(
        f"[fold] model={model_name} train={len(train_df)} val={len(val_df)} "
        f"batch={int(opt_cfg['batch_size'])} epochs={int(opt_cfg['epochs'])} "
        f"target_mean={target_stats['mean']} target_std={target_stats['std']}",
        flush=True,
    )
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, target_stats)
        va = run_epoch(model, val_loader, device, optimizer=None, target_stats=target_stats)
        row = {"epoch": epoch, "epoch_seconds": time.time() - t0, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(
            "[epoch {ep:03d}] sec={sec:.1f} train_loss={tl:.4f} train_mae=({ts:.3f},{td:.3f}) val_loss={vl:.4f} val_mae=({vs:.3f},{vd:.3f})".format(
                ep=epoch, sec=row["epoch_seconds"], tl=tr["loss"], ts=tr["mae_sbp"], td=tr["mae_dbp"], vl=va["loss"], vs=va["mae_sbp"], vd=va["mae_dbp"]
            ),
            flush=True,
        )
        score = va["mae_sbp"] + va["mae_dbp"]
        if score < best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            print(f"[epoch {epoch:03d}] new best score={score:.4f}", flush=True)
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] epoch={epoch} patience={patience}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    best_metrics = min(history, key=lambda r: r["val_mae_sbp"] + r["val_mae_dbp"])
    return {"history": history, "best": best_metrics, "state_dict": model.state_dict(), "target_stats": target_stats}


def evaluate_model(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    model_cfg: ModelConfig,
    device: torch.device,
    target_stats: dict[str, list[float]],
) -> dict[str, float]:
    runtime_cfg = cfg["runtime"]
    loader = DataLoader(
        ScgDataset(
            df,
            model_cfg.input_channels,
            model_cfg.window_size,
            cache_windows=bool(runtime_cfg.get("cache_windows", True)),
            mmap_arrays=bool(runtime_cfg.get("mmap_arrays", True)),
            allow_csv_fallback=bool(runtime_cfg.get("allow_csv_fallback", False)),
            target_stats=target_stats,
        ),
        batch_size=int(cfg["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(runtime_cfg["num_workers"]),
        pin_memory=bool(runtime_cfg.get("pin_memory", True)),
    )
    return run_epoch(model, loader, device, optimizer=None, target_stats=target_stats)


def predict_raw_model(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    model_cfg: ModelConfig,
    device: torch.device,
    target_stats: dict[str, list[float]],
) -> np.ndarray:
    runtime_cfg = cfg["runtime"]
    loader = DataLoader(
        ScgDataset(
            df,
            model_cfg.input_channels,
            model_cfg.window_size,
            cache_windows=bool(runtime_cfg.get("cache_windows", True)),
            mmap_arrays=bool(runtime_cfg.get("mmap_arrays", True)),
            allow_csv_fallback=bool(runtime_cfg.get("allow_csv_fallback", False)),
            target_stats=target_stats,
        ),
        batch_size=int(cfg["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(runtime_cfg["num_workers"]),
        pin_memory=bool(runtime_cfg.get("pin_memory", True)),
    )
    target_mean, target_std = _target_tensors(target_stats, device)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            pred = model(x) * target_std + target_mean
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros((0, 2), dtype=np.float32)


def _mae_np(pred: np.ndarray, df: pd.DataFrame) -> dict[str, float]:
    y = df[["SBP", "DBP"]].to_numpy(dtype=np.float32)
    err = np.abs(pred - y)
    return {
        "mae_sbp": float(err[:, 0].mean()) if len(err) else float("nan"),
        "mae_dbp": float(err[:, 1].mean()) if len(err) else float("nan"),
        "loss": float(err.mean()) if len(err) else float("nan"),
    }


def _prediction_frame(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = df[["sample_id", "subject_id", "SBP", "DBP"]].copy()
    out["pred_SBP"] = pred[:, 0]
    out["pred_DBP"] = pred[:, 1]
    out["abs_err_SBP"] = np.abs(out["pred_SBP"] - out["SBP"].astype(float))
    out["abs_err_DBP"] = np.abs(out["pred_DBP"] - out["DBP"].astype(float))
    return out


def _subject_error_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, g in pred_df.groupby("subject_id", sort=True):
        rows.append(
            {
                "subject_id": subject,
                "n": int(len(g)),
                "mae_sbp": float(g["abs_err_SBP"].mean()),
                "mae_dbp": float(g["abs_err_DBP"].mean()),
                "mae_mean": float((g["abs_err_SBP"].mean() + g["abs_err_DBP"].mean()) / 2),
                "true_sbp_mean": float(g["SBP"].mean()),
                "true_dbp_mean": float(g["DBP"].mean()),
                "pred_sbp_mean": float(g["pred_SBP"].mean()),
                "pred_dbp_mean": float(g["pred_DBP"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _device_from_cfg(runtime_cfg: dict[str, Any]) -> torch.device:
    device_name = runtime_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device_name != "cuda":
        return torch.device(device_name)
    gpu_ids = normalize_gpu_ids(runtime_cfg.get("gpu_ids", []))
    visible = torch.cuda.device_count()
    primary = gpu_ids[0] if gpu_ids and gpu_ids[0] < visible else 0
    torch.cuda.set_device(primary)
    return torch.device(f"cuda:{primary}")


def _model_cfg(cfg: dict[str, Any]) -> ModelConfig:
    mcfg = cfg["model"]
    return ModelConfig(
        input_channels=int(mcfg["input_channels"]),
        window_size=int(mcfg["window_size"]),
        cnn_channels=[int(x) for x in mcfg["cnn_channels"]],
        lstm_hidden=int(mcfg["lstm_hidden"]),
        lstm_layers=int(mcfg["lstm_layers"]),
        mlp_hidden=[int(x) for x in mcfg["mlp_hidden"]],
        dropout=float(mcfg["dropout"]),
    )


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _final_train_val_split(trainval: pd.DataFrame, folds: pd.DataFrame, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = folds[folds["fold"] == validation_fold][["sample_id", "subset"]]
    if fold_map.empty:
        raise ValueError(f"final_validation_fold={validation_fold} not found in folds.csv")
    merged = trainval.merge(fold_map, on="sample_id", how="inner")
    tr_df = merged[merged["subset"] == "train"].drop(columns=["subset"])
    va_df = merged[merged["subset"] == "val"].drop(columns=["subset"])
    if tr_df.empty or va_df.empty:
        raise RuntimeError(f"Invalid final validation fold {validation_fold}: train={len(tr_df)} val={len(va_df)}")
    tr_subjects = set(tr_df["subject_id"].astype(str))
    va_subjects = set(va_df["subject_id"].astype(str))
    overlap = tr_subjects & va_subjects
    if overlap:
        raise RuntimeError(f"Subject leakage in final validation fold {validation_fold}: {sorted(overlap)}")
    return tr_df, va_df


def _assert_no_label_group_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, context: str) -> None:
    if "label_group_id" not in train_df.columns or "label_group_id" not in val_df.columns:
        return
    overlap = set(train_df["label_group_id"].astype(str)) & set(val_df["label_group_id"].astype(str))
    if overlap:
        raise RuntimeError(f"Label-group leakage in {context}: {sorted(overlap)[:5]}")


def _final_train_val_split_for_cfg(
    trainval: pd.DataFrame,
    folds: pd.DataFrame,
    validation_fold: int,
    allow_subject_overlap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_df, va_df = _final_train_val_split_no_subject_check(trainval, folds, validation_fold)
    _assert_no_label_group_leakage(tr_df, va_df, f"final validation fold {validation_fold}")
    if not allow_subject_overlap:
        tr_subjects = set(tr_df["subject_id"].astype(str))
        va_subjects = set(va_df["subject_id"].astype(str))
        overlap = tr_subjects & va_subjects
        if overlap:
            raise RuntimeError(f"Subject leakage in final validation fold {validation_fold}: {sorted(overlap)}")
    return tr_df, va_df


def _final_train_val_split_no_subject_check(trainval: pd.DataFrame, folds: pd.DataFrame, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_map = folds[folds["fold"] == validation_fold][["sample_id", "subset"]]
    if fold_map.empty:
        raise ValueError(f"final_validation_fold={validation_fold} not found in folds.csv")
    merged = trainval.merge(fold_map, on="sample_id", how="inner")
    tr_df = merged[merged["subset"] == "train"].drop(columns=["subset"])
    va_df = merged[merged["subset"] == "val"].drop(columns=["subset"])
    if tr_df.empty or va_df.empty:
        raise RuntimeError(f"Invalid final validation fold {validation_fold}: train={len(tr_df)} val={len(va_df)}")
    return tr_df, va_df


def run(model_name: str, config_path: str, overrides: list[str] | None = None, fold_filter: int | None = None, mode: str = "all") -> None:
    cfg = load_with_overrides(config_path, overrides)
    runtime_cfg = cfg["runtime"]
    set_seed(int(runtime_cfg["seed"]))
    device = _device_from_cfg(runtime_cfg)
    split_dir = Path(cfg["input"]["split_dir"])
    trainval = pd.read_csv(split_dir / "trainval.csv")
    folds = pd.read_csv(split_dir / "folds.csv")
    test_df = pd.read_csv(split_dir / "test.csv")
    model_cfg = _model_cfg(cfg)
    tag_parts = [now_tag(), model_name]
    if fold_filter is not None:
        tag_parts.append(f"fold{fold_filter}")
    elif mode == "final":
        tag_parts.append("final")
    run_dir = ensure_dir(Path(cfg["output"]["runs_dir"]) / "_".join(tag_parts))
    save_json(run_dir / "config.resolved.json", cfg)
    print(
        f"[start] model={model_name} mode={mode} fold={fold_filter} device={device} cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()} trainval={len(trainval)} test={len(test_df)} run_dir={run_dir}",
        flush=True,
    )

    start_time = time.time()
    fold_metrics: list[dict[str, Any]] = []
    cv_states: list[dict[str, Any]] = []
    if mode in {"all", "cv"}:
        unique_folds = sorted(int(x) for x in folds["fold"].unique().tolist())
        if fold_filter is not None:
            unique_folds = [fold_filter]
        for fold in unique_folds:
            print(f"[cv] fold={fold}/{len(unique_folds)}", flush=True)
            fold_map = folds[folds["fold"] == fold][["sample_id", "subset"]]
            merged = trainval.merge(fold_map, on="sample_id", how="inner")
            tr_df = merged[merged["subset"] == "train"].drop(columns=["subset"])
            va_df = merged[merged["subset"] == "val"].drop(columns=["subset"])
            _assert_no_label_group_leakage(tr_df, va_df, f"cv fold {fold}")
            result = fit_one_fold(model_name, model_cfg, tr_df, va_df, cfg, device)
            best = result["best"]
            fold_metrics.append({"model": model_name, "fold": int(fold), "val_mae_sbp": float(best["val_mae_sbp"]), "val_mae_dbp": float(best["val_mae_dbp"]), "val_loss": float(best["val_loss"]), "run_dir": str(run_dir)})
            cv_states.append({"fold": int(fold), "state_dict": result["state_dict"], "target_stats": result["target_stats"]})
            fold_dir = ensure_dir(run_dir / f"fold_{int(fold)}")
            _write_history(fold_dir / "metrics.csv", result["history"])
            torch.save(result["state_dict"], fold_dir / "best.pt")
            save_json(fold_dir / "target_stats.json", result["target_stats"])
            print(f"[cv] fold={fold} best_val_mae=({fold_metrics[-1]['val_mae_sbp']:.3f},{fold_metrics[-1]['val_mae_dbp']:.3f})", flush=True)

    test_metrics: dict[str, float] = {"mae_sbp": float("nan"), "mae_dbp": float("nan"), "loss": float("nan")}
    final_validation_fold = int(cfg["optimization"].get("final_validation_fold", 1))
    test_strategy = str(cfg["optimization"].get("test_strategy", "final"))
    ensemble_test_metrics: dict[str, float] | None = None
    if mode == "all" and fold_filter is None and test_strategy in {"cv_ensemble", "both"} and cv_states:
        fold_preds: list[np.ndarray] = []
        for item in cv_states:
            fold_model = build_model(model_name, model_cfg).to(device)
            fold_model.load_state_dict(item["state_dict"])
            fold_preds.append(predict_raw_model(fold_model, test_df, cfg, model_cfg, device, item["target_stats"]))
        ensemble_pred = np.mean(np.stack(fold_preds, axis=0), axis=0)
        ensemble_test_metrics = _mae_np(ensemble_pred, test_df)
        pred_df = _prediction_frame(test_df, ensemble_pred)
        pred_df.to_csv(run_dir / "cv_ensemble_test_predictions.csv", index=False)
        _subject_error_frame(pred_df).to_csv(run_dir / "cv_ensemble_subject_errors.csv", index=False)
        print(
            "[cv-ensemble] test_mae=({sbp:.3f},{dbp:.3f}) folds={folds}".format(
                sbp=ensemble_test_metrics["mae_sbp"],
                dbp=ensemble_test_metrics["mae_dbp"],
                folds=len(cv_states),
            ),
            flush=True,
        )
        if test_strategy == "cv_ensemble":
            test_metrics = ensemble_test_metrics

    final_target_stats: dict[str, list[float]] | None = None
    if mode in {"all", "final"} and fold_filter is None and test_strategy != "cv_ensemble":
        final_train, final_val = _final_train_val_split_for_cfg(
            trainval,
            folds,
            final_validation_fold,
            allow_subject_overlap=bool(cfg["optimization"].get("allow_subject_overlap_validation", False)),
        )
        print(
            f"[final] training with internal validation fold={final_validation_fold} "
            f"train={len(final_train)} val={len(final_val)} test={len(test_df)}",
            flush=True,
        )
        final_result = fit_one_fold(model_name, model_cfg, final_train, final_val, cfg, device)
        final_target_stats = final_result["target_stats"]
        final_model = build_model(model_name, model_cfg).to(device)
        final_model.load_state_dict(final_result["state_dict"])
        test_metrics = evaluate_model(final_model, test_df, cfg, model_cfg, device, final_target_stats)
        final_dir = ensure_dir(run_dir / "final")
        _write_history(final_dir / "metrics.csv", final_result["history"])
        torch.save(final_result["state_dict"], final_dir / "best.pt")
        save_json(final_dir / "target_stats.json", final_target_stats)

    if fold_metrics:
        pd.DataFrame(fold_metrics).to_csv(run_dir / "fold_metrics.csv", index=False)
    summary = {
        "model": model_name,
        "mode": mode,
        "fold": fold_filter,
        "device": str(device),
        "run_dir": str(run_dir),
        "folds": len(fold_metrics),
        "cv_val_mae_sbp": float(np.mean([r["val_mae_sbp"] for r in fold_metrics])) if fold_metrics else float("nan"),
        "cv_val_mae_dbp": float(np.mean([r["val_mae_dbp"] for r in fold_metrics])) if fold_metrics else float("nan"),
        "test_mae_sbp": float(test_metrics["mae_sbp"]),
        "test_mae_dbp": float(test_metrics["mae_dbp"]),
        "test_loss": float(test_metrics["loss"]),
        "runtime_seconds": float(time.time() - start_time),
        "target_standardize": bool(cfg["optimization"].get("target_standardize", True)),
        "final_validation_fold": final_validation_fold if mode in {"all", "final"} and fold_filter is None else None,
        "test_strategy": test_strategy,
        "ensemble_test_mae_sbp": float(ensemble_test_metrics["mae_sbp"]) if ensemble_test_metrics else float("nan"),
        "ensemble_test_mae_dbp": float(ensemble_test_metrics["mae_dbp"]) if ensemble_test_metrics else float("nan"),
        "final_target_stats": final_target_stats,
        "model_config": asdict(model_cfg),
    }
    save_json(run_dir / "metrics.json", summary)
    print(f"[done] {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one SCG-BP model variant.")
    parser.add_argument("--model", required=True, choices=["full", "cnn_only", "lstm_only", "mlp_only"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fold", type=int, default=None, help="Run only one CV fold.")
    parser.add_argument("--mode", choices=["all", "cv", "final"], default="all")
    args = parser.parse_args()
    mode = "cv" if args.fold is not None and args.mode == "all" else args.mode
    run(args.model, args.config, args.override, fold_filter=args.fold, mode=mode)


if __name__ == "__main__":
    main()
