from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from scg_bp.calibrated_split import _subject_group_holdout, _with_label_group
from scg_bp.io_bp import detect_bp_format, standardize_bp


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


if __name__ == "__main__":
    unittest.main()
