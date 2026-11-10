"""Chronological benchmark regression tests."""
import unittest

import numpy as np
import pandas as pd

from capabilities.benchmark import score, rolling_benchmark, aggregate_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_metric_values(self):
        metrics = score([100, 200], [90, 220])
        self.assertAlmostEqual(metrics["mape"], 10.0)
        self.assertAlmostEqual(metrics["wape"], 10.0)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(250))

    def test_zero_actual_handling(self):
        metrics = score([0, 100], [5, 90])
        self.assertAlmostEqual(metrics["mape"], 10.0)
        self.assertAlmostEqual(metrics["wape"], 15.0)
        self.assertIsNone(score([0, 0], [1, 2])["wape"])

    def test_chronological_folds_and_failure_reporting(self):
        ts = pd.Series(np.arange(60, dtype=float) + 100,
                       index=pd.date_range("2026-01-01", periods=60))
        seen = []

        def baseline(train, horizon):
            seen.append((len(train), train.index.max()))
            return np.repeat(train.iloc[-1], horizon)

        def broken(train, horizon):
            raise RuntimeError("intentional")

        rows = rolling_benchmark(ts, {"baseline": baseline, "broken": broken},
                                 horizon=5, folds=3, min_train=30)
        self.assertEqual([n for n, _ in seen], [45, 50, 55])
        self.assertEqual(len(rows), 6)
        aggregate = {r["model"]: r for r in aggregate_benchmark(rows)}
        self.assertEqual(aggregate["baseline"]["failed_folds"], 0)
        self.assertEqual(aggregate["broken"]["failed_folds"], 3)
        self.assertIsNone(aggregate["broken"]["wape"])

    def test_insufficient_data_rejected(self):
        ts = pd.Series(np.arange(10, dtype=float),
                       index=pd.date_range("2026-01-01", periods=10))
        with self.assertRaises(ValueError):
            rolling_benchmark(ts, {}, horizon=7, folds=3)


if __name__ == "__main__":
    unittest.main()
