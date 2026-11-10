"""Prove winsorization is fitted on training only and test actuals stay raw."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from capabilities.forecasting import prepare_series, run_backtest


class FoldWinsorizationTests(unittest.TestCase):
    def test_preparation_never_clips_using_future_observations(self):
        ts = pd.DataFrame({
            "bill_date": pd.date_range("2026-01-01", periods=100),
            "billed_amount": [100.] * 99 + [100000.],
        })
        prepared = prepare_series(ts, "bill_date", "billed_amount",
                                  {"winsorize": True})
        self.assertEqual(prepared.iloc[-1], 100000.)

    def test_train_only_clipping_and_raw_test_scoring(self):
        index = pd.date_range("2026-01-01", periods=100)
        vals = np.full(100, 100.0)
        vals[70] = 1000.0
        vals[95:] = 100000.0
        ts = pd.Series(vals, index=index)
        seen = []

        def runner(train, horizon, **kwargs):
            seen.append(train.copy())
            return np.full(horizon, 100.0)

        model = SimpleNamespace(name="test_baseline", family="baseline",
                                available=True, runner=runner)
        with patch("capabilities.forecasting.build_registry", return_value=[model]):
            result = run_backtest(ts, ["test_baseline"], horizon=5, n_folds=2,
                                  preprocessing={"winsorize": True})["test_baseline"]
        self.assertEqual(len(seen), 3)  # two folds and final refit
        self.assertEqual(len(seen[0]), 90)
        self.assertEqual(len(seen[1]), 95)
        self.assertEqual(len(seen[2]), 100)
        # Later holdout spike cannot affect the first training fold.
        self.assertLess(seen[0].max(), 1000.)
        # Last-fold actuals are untouched, despite train clipping.
        self.assertTrue(np.all(result.backtest_actuals == 100000.))
        self.assertGreater(result.mape, 0)

    def test_future_outlier_does_not_change_earlier_fold_training(self):
        ts = pd.Series(np.arange(100, dtype=float) + 100,
                       index=pd.date_range("2026-01-01", periods=100))
        snapshots = []

        def runner(train, horizon, **kwargs):
            snapshots.append(train.to_numpy().copy())
            return np.full(horizon, 100.0)

        model = SimpleNamespace(name="baseline", family="baseline",
                                available=True, runner=runner)
        with patch("capabilities.forecasting.build_registry", return_value=[model]):
            run_backtest(ts, ["baseline"], horizon=5, n_folds=2,
                         preprocessing={"winsorize": True})
            original = snapshots[0].copy()
            snapshots.clear()
            changed = ts.copy()
            changed.iloc[-1] = 1000000.
            run_backtest(changed, ["baseline"], horizon=5, n_folds=2,
                         preprocessing={"winsorize": True})
        np.testing.assert_allclose(original, snapshots[0])


if __name__ == "__main__":
    unittest.main()
