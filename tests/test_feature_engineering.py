"""Regression tests for point-in-time rolling revenue features."""
import unittest

import pandas as pd

from stages.feature_engine import _time_features


class TimeFeatureTests(unittest.TestCase):
    def test_rolling_features_exclude_current_target(self):
        dates = pd.date_range("2026-01-01", periods=45, freq="D")
        baseline = pd.DataFrame({"date": dates, "revenue": [float(i) for i in range(45)]})
        changed = baseline.copy()
        changed.loc[20, "revenue"] = 1000000.0

        first, _ = _time_features(baseline, "date", "revenue", "daily")
        second, _ = _time_features(changed, "date", "revenue", "daily")

        for window in (7, 30):
            for statistic in ("rollmean", "rollstd"):
                column = f"revenue_{statistic}{window}"
                self.assertEqual(first.loc[20, column], second.loc[20, column],
                                 f"{column} leaked the current target")
        self.assertEqual(first.loc[20, "revenue_rollmean7"], 16.0)
        self.assertTrue(pd.isna(first.loc[0, "revenue_rollmean7"]))

    def test_lag_one_is_previous_observation(self):
        df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40),
                           "revenue": list(range(40))})
        result, _ = _time_features(df, "date", "revenue", "daily")
        self.assertEqual(result.loc[20, "revenue_lag1"], 19)


if __name__ == "__main__":
    unittest.main()
