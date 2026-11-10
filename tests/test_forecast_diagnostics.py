"""Read-only forecast diagnostic test with a tiny synthetic CSV."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.diagnose_forecast_csv import diagnose


class ForecastDiagnosticTests(unittest.TestCase):
    def test_three_scenarios_and_fold_boundaries(self):
        dates = pd.date_range("2025-01-01", periods=130)
        data = pd.DataFrame({"bill_date": dates, "billed_amount": 100 + np.arange(130)})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "revenue.csv"
            data.to_csv(path, index=False)
            result = diagnose(str(path), horizon=7, folds=3)
        self.assertEqual(result["observed_days"], 130)
        self.assertEqual(len(result["scenarios"]), 3)
        for scenario in result["scenarios"].values():
            self.assertEqual(len(scenario["folds"]), 3)
            self.assertGreaterEqual(scenario["pooled_wape"], 0)
            for fold in scenario["folds"]:
                self.assertLess(fold["train_end"], fold["test_start"])


if __name__ == "__main__":
    unittest.main()
