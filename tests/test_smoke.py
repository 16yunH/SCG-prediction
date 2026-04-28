from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
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


if __name__ == "__main__":
    unittest.main()
