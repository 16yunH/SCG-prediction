from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .config import load_with_overrides
from .utils import ensure_dir, save_json


def _group_holdout(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df.groupby("subject_id").size().reset_index(name="n")
    subjects = counts["subject_id"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(subjects)
    target = max(1, int(round(len(df) * test_size)))
    test_subjects: set[str] = set()
    total = 0
    count_map = dict(zip(counts["subject_id"].astype(str), counts["n"].astype(int)))
    for subject in subjects:
        if total >= target and test_subjects:
            break
        test_subjects.add(subject)
        total += count_map[subject]
    is_test = df["subject_id"].astype(str).isin(test_subjects)
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)


def _group_kfold_assignments(df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    counts = df.groupby("subject_id").size().reset_index(name="n").sort_values("n", ascending=False)
    n_splits = max(2, min(n_splits, len(counts)))
    fold_loads = [0 for _ in range(n_splits)]
    subject_to_fold: dict[str, int] = {}
    for _, row in counts.iterrows():
        fold_idx = min(range(n_splits), key=lambda i: fold_loads[i])
        subject = str(row["subject_id"])
        subject_to_fold[subject] = fold_idx + 1
        fold_loads[fold_idx] += int(row["n"])

    rows = []
    base = df[["sample_id", "subject_id"]].copy()
    base["assigned_fold"] = base["subject_id"].astype(str).map(subject_to_fold)
    for fold in range(1, n_splits + 1):
        fold_df = base[["sample_id", "subject_id"]].copy()
        fold_df["fold"] = fold
        fold_df["subset"] = "train"
        fold_df.loc[base["assigned_fold"] == fold, "subset"] = "val"
        rows.append(fold_df)
    return pd.concat(rows, ignore_index=True)


def _assert_no_leakage(trainval: pd.DataFrame, test: pd.DataFrame, folds: pd.DataFrame) -> None:
    train_subjects = set(trainval["subject_id"].astype(str))
    test_subjects = set(test["subject_id"].astype(str))
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"Subject leakage between trainval/test: {sorted(overlap)}")
    for fold, g in folds.groupby("fold"):
        tr = set(g[g["subset"] == "train"]["subject_id"].astype(str))
        va = set(g[g["subset"] == "val"]["subject_id"].astype(str))
        overlap = tr & va
        if overlap:
            raise RuntimeError(f"Subject leakage in fold {fold}: {sorted(overlap)}")


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    sample_index = pd.read_csv(cfg["input"]["sample_index"])
    required = {"sample_id", "subject_id"}
    missing = required - set(sample_index.columns)
    if missing:
        raise ValueError(f"sample_index missing columns: {sorted(missing)}")

    test_size = float(cfg["split"]["test_size"])
    seed = int(cfg["split"]["random_seed"])
    kfold_n = int(cfg["split"]["group_kfold"])
    trainval, test = _group_holdout(sample_index, test_size, seed)
    folds = _group_kfold_assignments(trainval, kfold_n)
    _assert_no_leakage(trainval, test, folds)

    split_dir = ensure_dir(cfg["output"]["split_dir"])
    trainval.to_csv(split_dir / "trainval.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    folds.to_csv(split_dir / "folds.csv", index=False)
    meta = {
        "trainval": len(trainval),
        "test": len(test),
        "train_subjects": int(trainval["subject_id"].nunique()),
        "test_subjects": int(test["subject_id"].nunique()),
        "kfold": int(folds["fold"].nunique()),
        "split_dir": str(split_dir),
        "splitter": "deterministic_group_holdout_plus_greedy_group_kfold",
    }
    save_json(split_dir / "split_meta.json", meta)
    print(meta, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subject-level splits and GroupKFold-style partitions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
