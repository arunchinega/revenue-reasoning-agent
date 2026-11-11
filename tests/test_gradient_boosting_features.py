"""Gradient Boosting forecasting integration regression tests."""
import unittest
import numpy as np
import pandas as pd

from capabilities.forecasting import _run_gradient_boosting


class GradientBoostingFeatureTests(unittest.TestCase):
    def test_calendar_aware_forecast_is_finite_and_repeatable(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        values = np.array([100 + 8 * (d.dayofweek < 5) + d.month for d in dates], dtype=float)
        series = pd.Series(values, index=dates)
        a = _run_gradient_boosting(series, horizon=7)
        b = _run_gradient_boosting(series, horizon=7)
        self.assertEqual(a.shape, (7,))
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a, b)

    def test_insufficient_history_fails_closed(self):
        series = pd.Series([1.0, 2.0, 3.0],
                           index=pd.date_range("2026-01-01", periods=3))
        with self.assertRaises(ValueError):
            _run_gradient_boosting(series, horizon=2)


if __name__ == "__main__":
    unittest.main()
