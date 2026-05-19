from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from scg_bp.data.io_bp import detect_bp_format, standardize_bp
from scg_bp.data.windowing import build_window_index, repair_bp_time_tokens
from scg_bp.splits.calibrated import _subject_group_holdout, _with_label_group
from scg_bp.splits.standard import _filter_supervised_samples


class SmokeTests(unittest.TestCase):
    def test_detect_text_csv(self) -> None:
        p = ROOT / "tests" / "fixtures" / "BP_text.csv"
        self.assertEqual(detect_bp_format(p), "text_csv")

    def test_standardize_bp(self) -> None:
        df = pd.DataFrame({"t": [61030, 61035], "SBP": [120, 121], "DBP": [80, 81]})
        out = standardize_bp(df, "001", "sess", Path("BP.xlsx"), "xlsx")
        self.assertEqual(len(out), 2)
        self.assertEqual(out.loc[0, "bp_time_token"], "061030")

    def test_calibrated_holdout_keeps_label_groups_separate(self) -> None:
        rows = []
        for subject in ["s1", "s2"]:
            for bp_idx in range(5):
                for jitter in range(3):
                    rows.append(
                        {
                            "sample_id": f"{subject}_{bp_idx}_{jitter}",
                            "subject_id": subject,
                            "session_id": "day1",
                            "bp_time_token": f"{bp_idx:06d}",
                            "SBP": 120 + bp_idx,
                            "DBP": 70 + bp_idx,
                            "HR": 60,
                        }
                    )
        df = _with_label_group(pd.DataFrame(rows))
        train, test = _subject_group_holdout(df, test_size=0.4, seed=1, min_groups_per_subject=3)
        self.assertEqual(set(train["subject_id"]), {"s1", "s2"})
        self.assertEqual(set(test["subject_id"]), {"s1", "s2"})
        self.assertFalse(set(train["label_group_id"]) & set(test["label_group_id"]))

    def test_repair_invalid_bp_time_token_by_neighbors(self) -> None:
        df = pd.DataFrame(
            {
                "subject_id": ["s1", "s1", "s1"],
                "session_id": ["day1", "day1", "day1"],
                "bp_row_index": [0, 1, 2],
                "bp_time_token": ["132151", "13256", "132201"],
                "SBP": [110, 111, 112],
                "DBP": [70, 71, 72],
                "HR": [60, 61, 62],
            }
        )
        out, corrections = repair_bp_time_tokens(df)
        self.assertEqual(out.loc[1, "bp_time_token"], "132156")
        self.assertEqual(out.loc[1, "bp_time_status"], "repaired")
        self.assertEqual(len(corrections), 1)

    def test_v3_window_generation_adds_interpolated_and_unlabeled_rows(self) -> None:
        bp = pd.DataFrame(
            {
                "subject_id": ["s1", "s1"],
                "session_id": ["day1", "day1"],
                "bp_row_index": [0, 1],
                "bp_time_token": ["010000", "010010"],
                "bp_time_minutes": [1440, 1450],
                "label_group_id": ["g0", "g1"],
                "SBP": [100.0, 120.0],
                "DBP": [60.0, 80.0],
                "HR": [70.0, 72.0],
                "PP": [40.0, 40.0],
            }
        )
        sig = pd.DataFrame(
            {
                "signal_id": ["sig_001"],
                "subject_id": ["s1"],
                "session_id": ["day1"],
                "source_file": ["dummy.csv"],
                "array_path": ["dummy.npy"],
                "n_rows": [3000],
                "array_rows": [3000],
                "n_cols": [7],
                "channel_mode": ["7col"],
            }
        )
        windows, _ = build_window_index(
            bp,
            sig,
            sample_rate_hz=50,
            window_seconds=8,
            alignment_method="bp_time_interpolation",
            stride_seconds=2,
            jitter_steps=0,
            window_cfg={
                "measured_seconds": [8],
                "measured_offsets_sec": [0],
                "interpolated": {"enabled": True, "stride_seconds": 60, "exclude_margin_seconds": 60, "label_weight": 0.2},
                "unlabeled": {"enabled": True, "seconds": [8], "stride_seconds": 30},
            },
        )
        counts = windows["label_source"].value_counts().to_dict()
        self.assertEqual(counts["measured_bp"], 2)
        self.assertEqual(counts["interpolated_bp"], 9)
        self.assertGreater(counts["unlabeled"], 0)
        interp = windows[windows["label_source"] == "interpolated_bp"].iloc[0]
        self.assertEqual(interp["left_label_group_id"], "g0")
        self.assertEqual(interp["right_label_group_id"], "g1")
        self.assertAlmostEqual(float(interp["label_weight"]), 0.2)

    def test_split_filter_excludes_unlabeled_rows(self) -> None:
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "subject_id": ["s1", "s1", "s1"],
                "label_source": ["measured_bp", "interpolated_bp", "unlabeled"],
                "is_supervised": [True, True, False],
                "SBP": [100.0, 101.0, pd.NA],
                "DBP": [70.0, 71.0, pd.NA],
            }
        )
        out = _filter_supervised_samples(df, {"input": {"include_label_sources": ["measured_bp", "interpolated_bp"]}})
        self.assertEqual(out["sample_id"].tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
