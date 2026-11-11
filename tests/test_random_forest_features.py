"""Random Forest forecasting feature integration tests."""
import unittest
import numpy as np
import pandas as pd

from capabilities.forecasting import _run_random_forest


class RandomForestFeatureTests(unittest.TestCase):
    def test_calendar_forecast_is_finite_and_repeatable(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        series = pd.Series([100.0 + 10.0 * (d.dayofweek < 5) + d.month for d in dates],
                           index=dates)
        first = _run_random_forest(series, horizon=7)
        second = _run_random_forest(series, horizon=7)
        self.assertEqual(first.shape, (7,))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, second)

    def test_short_series_fails_closed(self):
        series = pd.Series([1.0, 2.0, 3.0],
                           index=pd.date_range("2026-01-01", periods=3))
        with self.assertRaises(ValueError):
            _run_random_forest(series, horizon=2)


if __name__ == "__main__":
    unittest.main()
