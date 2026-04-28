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
from .io_scg import read_scg_window
from .models import ModelConfig, build_model
from .utils import ensure_dir, now_tag, save_json, set_seed


class ScgDataset(Dataset):
    def __init__(self, sample_df: pd.DataFrame, input_channels: int, window_size: int) -> None:
        self.df = sample_df.reset_index(drop=True)
        self.input_channels = input_channels
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
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
        x = x.to(device)
        y = y.to(device)

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


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, batch_size: int, workers: int, channels: int, window_size: int) -> tuple[DataLoader, DataLoader]:
    train_ds = ScgDataset(train_df, channels, window_size)
    val_ds = ScgDataset(val_df, channels, window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader


def fit_one_fold(
    model_name: str,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(model_name, model_cfg).to(device)
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
        workers=int(cfg["runtime"]["num_workers"]),
        channels=model_cfg.input_channels,
        window_size=model_cfg.window_size,
    )

    best_val = float("inf")
    best_state = None
    patience = int(opt_cfg["early_stop_patience"])
    bad = 0

    history: list[dict[str, float]] = []
    for epoch in range(1, int(opt_cfg["epochs"]) + 1):
        tr = run_epoch(model, train_loader, device, optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)

        score = va["mae_sbp"] + va["mae_dbp"]
        if score < best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    best_metrics = min(history, key=lambda r: r["val_mae_sbp"] + r["val_mae_dbp"])
    return {"history": history, "best": best_metrics, "state_dict": model.state_dict()}


def evaluate_model(model: nn.Module, df: pd.DataFrame, cfg: dict[str, Any], model_cfg: ModelConfig, device: torch.device) -> dict[str, float]:
    loader = DataLoader(
        ScgDataset(df, model_cfg.input_channels, model_cfg.window_size),
        batch_size=int(cfg["optimization"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["runtime"]["num_workers"]),
    )
    return run_epoch(model, loader, device, optimizer=None)


def run(model_name: str, config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    set_seed(int(cfg["runtime"]["seed"]))

    device_name = cfg["runtime"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
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
    fold_metrics = []

    unique_folds = sorted(folds["fold"].unique().tolist())
    for fold in unique_folds:
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

    # Final model on full trainval, validated on holdout test for report.
    final_result = fit_one_fold(model_name, model_cfg, trainval, test_df, cfg, device)
    final_model = build_model(model_name, model_cfg).to(device)
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

    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model variant.")
    parser.add_argument("--model", required=True, choices=["full", "cnn_only", "lstm_only", "mlp_only"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.model, args.config, args.override)


if __name__ == "__main__":
    main()
