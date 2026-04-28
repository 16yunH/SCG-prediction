from __future__ import annotations

import argparse
import csv
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


class ScgDataset(Dataset):
    def __init__(self, sample_df: pd.DataFrame, input_channels: int, window_size: int, cache_windows: bool = True) -> None:
        self.df = sample_df.reset_index(drop=True)
        self.input_channels = input_channels
        self.window_size = window_size
        self.cache_windows = cache_windows
        self.x_cache: list[torch.Tensor] | None = None
        self.y_cache: list[torch.Tensor] | None = None
        if self.cache_windows:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _row_to_xy(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = Path(row["scg_file"])
        mode = str(row["scg_mode"])

        win = read_scg_window(path, int(row["start_row"]), int(row["end_row"]), mode)
        arr = win.to_numpy(dtype=np.float32)

        if arr.shape[1] < self.input_channels:
            pad = np.zeros((arr.shape[0], self.input_channels - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        arr = arr[:, : self.input_channels]

        if arr.shape[0] < self.window_size:
            pad = np.zeros((self.window_size - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        arr = arr[: self.window_size, :]

        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-6
        arr = (arr - mean) / std

        x = torch.from_numpy(arr.T)  # [C, T]
        y = torch.tensor([float(row["SBP"]), float(row["DBP"])], dtype=torch.float32)
        return x, y

    def _build_cache(self) -> None:
        self.x_cache = []
        self.y_cache = []
        file_cache: dict[tuple[str, str], np.ndarray] = {}
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            key = (str(row["scg_file"]), str(row["scg_mode"]))
            if key not in file_cache:
                file_cache[key] = read_scg_full_array(Path(key[0]), key[1], self.input_channels)

            full = file_cache[key]
            start_row = int(row["start_row"])
            end_row = int(row["end_row"])
            start_row = max(0, min(start_row, max(0, full.shape[0] - 1)))
            end_row = max(start_row + 1, min(end_row, full.shape[0]))
            arr = full[start_row:end_row]

            if arr.shape[0] < self.window_size:
                pad = np.zeros((self.window_size - arr.shape[0], arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            arr = arr[: self.window_size, :]

            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-6
            arr = (arr - mean) / std

            x = torch.from_numpy(arr.T)
            y = torch.tensor([float(row["SBP"]), float(row["DBP"])], dtype=torch.float32)
            self.x_cache.append(x)
            self.y_cache.append(y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.x_cache is not None and self.y_cache is not None:
            return self.x_cache[idx], self.y_cache[idx]
        return self._row_to_xy(idx)


def mae_per_target(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    e = torch.abs(pred - target)
    sbp = float(e[:, 0].mean().item())
    dbp = float(e[:, 1].mean().item())
    return sbp, dbp


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    loss_fn = nn.SmoothL1Loss()
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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        sbp, dbp = mae_per_target(pred.detach(), y)
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
) -> tuple[DataLoader, DataLoader]:
    train_ds = ScgDataset(train_df, channels, window_size, cache_windows=cache_windows)
    val_ds = ScgDataset(val_df, channels, window_size, cache_windows=cache_windows)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def fit_one_fold(
    model_name: str,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    runtime_cfg = cfg["runtime"]
    model = build_model(model_name, model_cfg).to(device)
    gpu_ids = normalize_gpu_ids(runtime_cfg.get("gpu_ids", []))
    if (
        device.type == "cuda"
        and bool(runtime_cfg.get("use_data_parallel", False))
        and len(gpu_ids) > 1
    ):
        model = nn.DataParallel(model, device_ids=gpu_ids)
    opt_cfg = cfg["optimization"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )

    train_loader, val_loader = build_loaders(
        train_df=train_df,
        val_df=val_df,
        batch_size=int(opt_cfg["batch_size"]),
        workers=int(runtime_cfg["num_workers"]),
        channels=model_cfg.input_channels,
        window_size=model_cfg.window_size,
        pin_memory=bool(runtime_cfg.get("pin_memory", True)),
        cache_windows=bool(runtime_cfg.get("cache_windows", True)),
    )

    best_val = float("inf")
    best_state = None
    patience = int(opt_cfg["early_stop_patience"])
    bad = 0

    history: list[dict[str, float]] = []
    print(
        f"[fold] model={model_name} train={len(train_df)} val={len(val_df)} "
        f"batch={int(opt_cfg['batch_size'])} epochs={int(opt_cfg['epochs'])}",
        flush=True,
    )
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(
            "[epoch {ep:03d}] train_loss={tl:.4f} train_mae=({ts:.3f},{td:.3f}) "
            "val_loss={vl:.4f} val_mae=({vs:.3f},{vd:.3f})".format(
                ep=epoch,
                tl=tr["loss"],
                ts=tr["mae_sbp"],
                td=tr["mae_dbp"],
                vl=va["loss"],
                vs=va["mae_sbp"],
                vd=va["mae_dbp"],
            ),
            flush=True,
        )

        score = va["mae_sbp"] + va["mae_dbp"]
        if score < best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
    return {"history": history, "best": best_metrics, "state_dict": model.state_dict()}


def evaluate_model(model: nn.Module, df: pd.DataFrame, cfg: dict[str, Any], model_cfg: ModelConfig, device: torch.device) -> dict[str, float]:
    runtime_cfg = cfg["runtime"]
    loader = DataLoader(
        ScgDataset(df, model_cfg.input_channels, model_cfg.window_size, cache_windows=bool(runtime_cfg.get("cache_windows", True))),
        batch_size=int(cfg["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(runtime_cfg["num_workers"]),
        pin_memory=bool(runtime_cfg.get("pin_memory", True)),
    )
    return run_epoch(model, loader, device, optimizer=None)


def run(model_name: str, config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    runtime_cfg = cfg["runtime"]
    set_seed(int(runtime_cfg["seed"]))
    gpu_ids = normalize_gpu_ids(runtime_cfg.get("gpu_ids", []))

    device_name = runtime_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "cuda":
        if gpu_ids:
            primary = gpu_ids[0]
            torch.cuda.set_device(primary)
            device = torch.device(f"cuda:{primary}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device(device_name)

    split_dir = Path(cfg["input"]["split_dir"])
    trainval = pd.read_csv(split_dir / "trainval.csv")
    folds = pd.read_csv(split_dir / "folds.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    mcfg = cfg["model"]
    model_cfg = ModelConfig(
        input_channels=int(mcfg["input_channels"]),
        window_size=int(mcfg["window_size"]),
        cnn_channels=[int(x) for x in mcfg["cnn_channels"]],
        lstm_hidden=int(mcfg["lstm_hidden"]),
        lstm_layers=int(mcfg["lstm_layers"]),
        mlp_hidden=[int(x) for x in mcfg["mlp_hidden"]],
        dropout=float(mcfg["dropout"]),
    )

    run_dir = ensure_dir(Path(cfg["output"]["runs_dir"]) / f"{now_tag()}_{model_name}")
    print(
        f"[start] model={model_name} device={device} cuda_available={torch.cuda.is_available()} "
        f"cuda_count={torch.cuda.device_count()} gpu_ids={gpu_ids} "
        f"data_parallel={runtime_cfg.get('use_data_parallel', False)} "
        f"trainval={len(trainval)} test={len(test_df)} run_dir={run_dir}",
        flush=True,
    )
    fold_metrics = []

    unique_folds = sorted(folds["fold"].unique().tolist())
    for fold in unique_folds:
        print(f"[cv] fold={fold}/{len(unique_folds)}", flush=True)
        fold_map = folds[folds["fold"] == fold][["sample_id", "subset"]]
        merged = trainval.merge(fold_map, on="sample_id", how="inner")
        tr_df = merged[merged["subset"] == "train"].drop(columns=["subset"])
        va_df = merged[merged["subset"] == "val"].drop(columns=["subset"])

        result = fit_one_fold(model_name, model_cfg, tr_df, va_df, cfg, device)
        best = result["best"]
        fold_metrics.append(
            {
                "model": model_name,
                "fold": int(fold),
                "val_mae_sbp": float(best["val_mae_sbp"]),
                "val_mae_dbp": float(best["val_mae_dbp"]),
                "val_loss": float(best["val_loss"]),
            }
        )

        hist_path = run_dir / f"fold_{int(fold)}_history.csv"
        with hist_path.open("w", newline="", encoding="utf-8") as f:
            if result["history"]:
                writer = csv.DictWriter(f, fieldnames=list(result["history"][0].keys()))
                writer.writeheader()
                writer.writerows(result["history"])
        print(
            f"[cv] fold={fold} best_val_mae=({fold_metrics[-1]['val_mae_sbp']:.3f},{fold_metrics[-1]['val_mae_dbp']:.3f})",
            flush=True,
        )

    # Final model on full trainval, validated on holdout test for report.
    print("[final] training on trainval and selecting by holdout metric pass", flush=True)
    final_result = fit_one_fold(model_name, model_cfg, trainval, test_df, cfg, device)
    final_model = build_model(model_name, model_cfg).to(device)
    if (
        device.type == "cuda"
        and bool(runtime_cfg.get("use_data_parallel", False))
        and len(gpu_ids) > 1
    ):
        final_model = nn.DataParallel(final_model, device_ids=gpu_ids)
    final_model.load_state_dict(final_result["state_dict"])
    test_metrics = evaluate_model(final_model, test_df, cfg, model_cfg, device)

    torch.save(final_result["state_dict"], run_dir / "model.pt")

    summary = {
        "model": model_name,
        "device": device.type,
        "run_dir": str(run_dir),
        "folds": len(unique_folds),
        "cv_val_mae_sbp": float(np.mean([r["val_mae_sbp"] for r in fold_metrics])),
        "cv_val_mae_dbp": float(np.mean([r["val_mae_dbp"] for r in fold_metrics])),
        "test_mae_sbp": float(test_metrics["mae_sbp"]),
        "test_mae_dbp": float(test_metrics["mae_dbp"]),
        "test_loss": float(test_metrics["loss"]),
        "model_config": asdict(model_cfg),
    }

    pd.DataFrame(fold_metrics).to_csv(run_dir / "fold_metrics.csv", index=False)
    save_json(run_dir / "metrics.json", summary)

    print(f"[done] {summary}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model variant.")
    parser.add_argument("--model", required=True, choices=["full", "cnn_only", "lstm_only", "mlp_only"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.model, args.config, args.override)


if __name__ == "__main__":
    main()
