from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .config import load_with_overrides
from .utils import ensure_dir


def run(config_path: str, overrides: list[str] | None = None) -> None:
    cfg = load_with_overrides(config_path, overrides)
    sample_index = pd.read_csv(cfg["input"]["sample_index"])

    if "subject_id" not in sample_index.columns:
        raise ValueError("sample_index missing subject_id")

    test_size = float(cfg["split"]["test_size"])
    seed = int(cfg["split"]["random_seed"])
    kfold_n = int(cfg["split"]["group_kfold"])

    groups = sample_index["subject_id"].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(sample_index, groups=groups))

    trainval = sample_index.iloc[train_idx].reset_index(drop=True)
    test = sample_index.iloc[test_idx].reset_index(drop=True)

    gkf = GroupKFold(n_splits=min(kfold_n, trainval["subject_id"].nunique()))
    fold_rows = []
    for fold, (tr, va) in enumerate(gkf.split(trainval, groups=trainval["subject_id"].astype(str).values), start=1):
        tr_ids = set(trainval.iloc[tr]["sample_id"].tolist())
        va_ids = set(trainval.iloc[va]["sample_id"].tolist())

        fold_df = trainval[["sample_id", "subject_id"]].copy()
        fold_df["fold"] = fold
        fold_df["subset"] = "train"
        fold_df.loc[fold_df["sample_id"].isin(va_ids), "subset"] = "val"
        fold_df.loc[fold_df["sample_id"].isin(tr_ids), "subset"] = "train"
        fold_rows.append(fold_df)

    folds = pd.concat(fold_rows, ignore_index=True)

    split_dir = ensure_dir(cfg["output"]["split_dir"])
    trainval.to_csv(split_dir / "trainval.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)
    folds.to_csv(split_dir / "folds.csv", index=False)

    print(
        {
            "trainval": len(trainval),
            "test": len(test),
            "train_subjects": trainval["subject_id"].nunique(),
            "test_subjects": test["subject_id"].nunique(),
            "kfold": int(folds["fold"].nunique()),
            "split_dir": str(split_dir),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subject-level splits and GroupKFold partitions.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    run(args.config, args.override)


if __name__ == "__main__":
    main()
