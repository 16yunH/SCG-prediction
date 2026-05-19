from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import load_with_overrides
from ..data.windowing import label_group_id
from ..utils import ensure_dir, save_json
from .standard import _filter_supervised_samples


def _with_label_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "label_group_id" not in out.columns:
        out["label_group_id"] = out.apply(label_group_id, axis=1)
    if "interval_group_id" not in out.columns:
        out["interval_group_id"] = out["label_group_id"]
    if "label_source" not in out.columns:
        out["label_source"] = "measured_bp"
    return out


def _subject_group_holdout(df: pd.DataFrame, test_size: float, seed: int, min_groups_per_subject: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for subject, g in df.groupby("subject_id", sort=True):
        groups = sorted(g["label_group_id"].astype(str).unique().tolist())
        rng.shuffle(groups)
        if len(groups) < min_groups_per_subject:
            train_parts.append(g)
            continue
        n_test = max(1, int(round(len(groups) * test_size)))
        n_test = min(n_test, len(groups) - 1)
        test_groups = set(groups[:n_test])
        is_test = g["label_group_id"].astype(str).isin(test_groups)
        test_parts.append(g.loc[is_test])
        train_parts.append(g.loc[~is_test])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train, test


def _subject_group_kfold(df: pd.DataFrame, n_splits: int, seed: int, min_groups_per_subject: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[pd.DataFrame] = []
    base = df[["sample_id", "subject_id", "label_group_id"]].copy()
    fold_maps: dict[str, int] = {}
    for subject, g in base.groupby("subject_id", sort=True):
        groups = sorted(g["label_group_id"].astype(str).unique().tolist())
        rng.shuffle(groups)
        if len(groups) < min_groups_per_subject:
            for group in groups:
                fold_maps[group] = 0
            continue
        subject_splits = max(2, min(n_splits, len(groups)))
        for idx, group in enumerate(groups):
            fold_maps[group] = (idx % subject_splits) + 1

    assigned = base["label_group_id"].astype(str).map(fold_maps).fillna(0).astype(int)
    for fold in range(1, n_splits + 1):
        fold_df = base[["sample_id", "subject_id"]].copy()
        fold_df["fold"] = fold
        fold_df["subset"] = "train"
        fold_df.loc[assigned == fold, "subset"] = "val"
        rows.append(fold_df)
    return pd.concat(rows, ignore_index=True)


def _assert_group_integrity(trainval: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame) -> None:
    for group_col in [c for c in ["label_group_id", "interval_group_id"] if c in trainval.columns and c in test.columns]:
        train_groups = set(trainval[group_col].astype(str))
        test_groups = set(test[group_col].astype(str))
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"{group_col} leakage between trainval/test: {sorted(overlap)[:5]}")
    group_cols = [c for c in ["label_group_id", "interval_group_id"] if c in trainval.columns]
    group_lookup = trainval[["sample_id", *group_cols]]
    for fold, fold_map in folds.groupby("fold"):
        merged = fold_map.merge(group_lookup, on="sample_id", how="left")
        for group_col in group_cols:
            tr = set(merged[merged["subset"] == "train"][group_col].dropna().astype(str))
            va = set(merged[merged["subset"] == "val"][group_col].dropna().astype(str))
            overlap = tr & va
            if overlap:
                raise RuntimeError(f"{group_col} leakage in fold {fold}: {sorted(overlap)[:5]}")


def _has_interval_endpoints(df: pd.DataFrame) -> bool:
    return {"left_label_group_id", "right_label_group_id", "label_source"}.issubset(df.columns)


def _assign_interpolated_by_endpoint_groups(
    df: pd.DataFrame,
    train_groups: set[str],
    test_groups: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not _has_interval_endpoints(df):
        return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), 0
    interp = df[df["label_source"].astype(str) == "interpolated_bp"].copy()
    if interp.empty:
        return interp, interp, 0
    left = interp["left_label_group_id"].astype(str)
    right = interp["right_label_group_id"].astype(str)
    is_train = left.isin(train_groups) & right.isin(train_groups)
    is_test = left.isin(test_groups) & right.isin(test_groups)
    dropped = int((~is_train & ~is_test).sum())
    return interp.loc[is_train].copy(), interp.loc[is_test].copy(), dropped


def _subject_group_holdout_with_intervals(df: pd.DataFrame, test_size: float, seed: int, min_groups_per_subject: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    measured = df[df["label_source"].astype(str) != "interpolated_bp"].copy()
    train_measured, test_measured = _subject_group_holdout(measured, test_size, seed, min_groups_per_subject)
    train_groups = set(train_measured["label_group_id"].astype(str))
    test_groups = set(test_measured["label_group_id"].astype(str))
    train_interp, test_interp, dropped = _assign_interpolated_by_endpoint_groups(df, train_groups, test_groups)
    train = pd.concat([train_measured, train_interp], ignore_index=True)
    test = pd.concat([test_measured, test_interp], ignore_index=True)
    return train, test, dropped


def _subject_group_kfold_with_intervals(df: pd.DataFrame, n_splits: int, seed: int, min_groups_per_subject: int) -> tuple[pd.DataFrame, int]:
    measured = df[df["label_source"].astype(str) != "interpolated_bp"].copy()
    measured_folds = _subject_group_kfold(measured, n_splits, seed, min_groups_per_subject)
    group_lookup = measured[["sample_id", "label_group_id"]]
    fold_rows: list[pd.DataFrame] = []
    dropped = 0
    for fold, fold_map in measured_folds.groupby("fold"):
        merged = fold_map.merge(group_lookup, on="sample_id", how="left")
        train_groups = set(merged[merged["subset"] == "train"]["label_group_id"].dropna().astype(str))
        val_groups = set(merged[merged["subset"] == "val"]["label_group_id"].dropna().astype(str))
        train_interp, val_interp, fold_dropped = _assign_interpolated_by_endpoint_groups(df, train_groups, val_groups)
        dropped += fold_dropped
        interp_rows = []
        if not train_interp.empty:
            t = train_interp[["sample_id", "subject_id"]].copy()
            t["fold"] = fold
            t["subset"] = "train"
            interp_rows.append(t)
        if not val_interp.empty:
            v = val_interp[["sample_id", "subject_id"]].copy()
            v["fold"] = fold
            v["subset"] = "val"
            interp_rows.append(v)
        fold_rows.append(fold_map[["sample_id", "subject_id", "fold", "subset"]].copy())
        if interp_rows:
            fold_rows.append(pd.concat(interp_rows, ignore_index=True))
    return pd.concat(fold_rows, ignore_index=True), dropped


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    sample_index = _filter_supervised_samples(pd.read_csv(cfg["input"]["sample_index"]), cfg)
    required = {"sample_id", "subject_id", "session_id", "SBP", "DBP"}
    missing = required - set(sample_index.columns)
    if missing:
        raise ValueError(f"sample_index missing columns: {sorted(missing)}")

    df = _with_label_group(sample_index)
    split_cfg: dict[str, Any] = cfg["split"]
    seed = int(split_cfg["random_seed"])
    test_size = float(split_cfg["test_size"])
    kfold_n = int(split_cfg["group_kfold"])
    min_groups = int(split_cfg.get("min_groups_per_subject", 3))
    if _has_interval_endpoints(df) and (df["label_source"].astype(str) == "interpolated_bp").any():
        trainval, test, dropped_holdout_intervals = _subject_group_holdout_with_intervals(df, test_size, seed, min_groups)
        folds, dropped_fold_intervals = _subject_group_kfold_with_intervals(trainval, kfold_n, seed, min_groups)
    else:
        trainval, test = _subject_group_holdout(df, test_size, seed, min_groups)
        folds = _subject_group_kfold(trainval, kfold_n, seed, min_groups)
        dropped_holdout_intervals = 0
        dropped_fold_intervals = 0
    _assert_group_integrity(trainval, test, folds)

    split_dir = ensure_dir(cfg["output"]["split_dir"])
    trainval.to_csv(split_dir / "trainval.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    folds.to_csv(split_dir / "folds.csv", index=False)
    meta = {
        "trainval": int(len(trainval)),
        "test": int(len(test)),
        "train_subjects": int(trainval["subject_id"].nunique()),
        "test_subjects": int(test["subject_id"].nunique()),
        "train_label_groups": int(trainval["label_group_id"].nunique()),
        "test_label_groups": int(test["label_group_id"].nunique()),
        "kfold": int(folds["fold"].nunique()),
        "split_dir": str(split_dir),
        "splitter": "subject_dependent_label_group_holdout_and_kfold",
        "dropped_holdout_intervals": int(dropped_holdout_intervals),
        "dropped_fold_intervals": int(dropped_fold_intervals),
    }
    save_json(split_dir / "split_meta.json", meta)
    print(meta, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subject-dependent calibrated splits by BP label group.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
