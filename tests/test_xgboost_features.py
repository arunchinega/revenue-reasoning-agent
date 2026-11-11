"""XGBoost calendar-aware forecasting tests."""
import unittest
import numpy as np
import pandas as pd

from capabilities.forecasting import HAS_XGB, _run_xgboost_lags


@unittest.skipUnless(HAS_XGB, "XGBoost not installed")
class XGBoostFeatureTests(unittest.TestCase):
    def test_forecast_finite_and_repeatable(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        series = pd.Series([100.0 + 10.0 * (d.dayofweek < 5) + d.month for d in dates],
                           index=dates)
        a = _run_xgboost_lags(series, horizon=7)
        b = _run_xgboost_lags(series, horizon=7)
        self.assertEqual(a.shape, (7,))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a, b)

    def test_short_history_rejected(self):
        series = pd.Series([1.0, 2.0, 3.0],
                           index=pd.date_range("2026-01-01", periods=3))
        with self.assertRaises(ValueError):
            _run_xgboost_lags(series, horizon=2)


if __name__ == "__main__":
    unittest.main()
