from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir, save_json


def _label_group(row: pd.Series) -> str:
    parts = [
        str(row.get("subject_id", "")),
        str(row.get("session_id", "")),
        str(row.get("bp_time_token", "")),
        str(row.get("SBP", "")),
        str(row.get("DBP", "")),
        str(row.get("HR", "")),
    ]
    return "||".join(parts)


def _with_label_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label_group_id"] = out.apply(_label_group, axis=1)
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
    train_groups = set(trainval["label_group_id"].astype(str))
    test_groups = set(test["label_group_id"].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Label-group leakage between trainval/test: {sorted(overlap)[:5]}")
    group_lookup = trainval[["sample_id", "label_group_id"]]
    for fold, fold_map in folds.groupby("fold"):
        merged = fold_map.merge(group_lookup, on="sample_id", how="left")
        tr = set(merged[merged["subset"] == "train"]["label_group_id"].astype(str))
        va = set(merged[merged["subset"] == "val"]["label_group_id"].astype(str))
        overlap = tr & va
        if overlap:
            raise RuntimeError(f"Label-group leakage in fold {fold}: {sorted(overlap)[:5]}")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    sample_index = pd.read_csv(cfg["input"]["sample_index"])
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
    trainval, test = _subject_group_holdout(df, test_size, seed, min_groups)
    folds = _subject_group_kfold(trainval, kfold_n, seed, min_groups)
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
