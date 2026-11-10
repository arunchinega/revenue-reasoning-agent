"""Regression tests for point-in-time rolling revenue features."""
import unittest

import pandas as pd

from stages.feature_engine import _time_features, _customer_daily_features


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
                a, b = first.loc[20, column], second.loc[20, column]
                if pd.isna(a) and pd.isna(b):
                    continue  # insufficient history for this rolling window
                self.assertEqual(a, b, f"{column} leaked the current target")
        self.assertEqual(first.loc[20, "revenue_rollmean7"], 16.0)
        self.assertTrue(pd.isna(first.loc[0, "revenue_rollmean7"]))

    def test_lag_one_is_previous_observation(self):
        df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=40),
                           "revenue": list(range(40))})
        result, _ = _time_features(df, "date", "revenue", "daily")
        self.assertEqual(result.loc[20, "revenue_lag1"], 19)


    def test_customer_history_is_isolated_and_same_day_safe(self):
        dates = pd.date_range("2026-01-01", periods=9, freq="D")
        rows = []
        for day in dates:
            rows.extend([
                {"date": day, "customer_id": "A", "revenue": 10.0},
                {"date": day, "customer_id": "A", "revenue": 20.0},
                {"date": day, "customer_id": "B", "revenue": 100.0},
            ])
        df = pd.DataFrame(rows).sample(frac=1, random_state=4).reset_index(drop=True)
        result, _ = _customer_daily_features(df, "date", "customer_id", "revenue")
        a = result[(result.customer_id == "A") & (result.date == dates[8])]
        b = result[(result.customer_id == "B") & (result.date == dates[8])]
        self.assertEqual(a["customer_revenue_lag1"].tolist(), [30.0, 30.0])
        self.assertEqual(b["customer_revenue_lag1"].tolist(), [100.0])
        self.assertTrue((a["customer_revenue_rollmean7"] == 30.0).all())
        self.assertTrue((b["customer_revenue_rollmean7"] == 100.0).all())
        self.assertEqual(len(result), len(df))

    def test_current_day_mutation_does_not_change_its_features(self):
        dates = pd.date_range("2026-01-01", periods=9, freq="D")
        df = pd.DataFrame({"date": list(dates) * 2,
                           "customer_id": ["A"] * 9 + ["B"] * 9,
                           "revenue": [10.0] * 9 + [100.0] * 9})
        modified = df.copy()
        modified.loc[8, "revenue"] = 999999.0
        before, _ = _customer_daily_features(df, "date", "customer_id", "revenue")
        after, _ = _customer_daily_features(modified, "date", "customer_id", "revenue")
        for name in ("customer_revenue_lag1", "customer_revenue_rollmean7",
                     "customer_revenue_rollstd7"):
            self.assertEqual(before.loc[8, name], after.loc[8, name])
        self.assertEqual(before.loc[17, "customer_revenue_lag1"],
                         after.loc[17, "customer_revenue_lag1"])


if __name__ == "__main__":
    unittest.main()
