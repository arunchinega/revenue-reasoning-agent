"""Run a reproducible synthetic revenue benchmark (not production evidence).

Usage: python -m scripts.run_forecasting_benchmark
Outputs are written to benchmark_results/ (not committed automatically).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from capabilities.benchmark import rolling_benchmark, aggregate_benchmark
from capabilities.forecasting import (
    _run_seasonal_naive, _run_gradient_boosting, _run_random_forest,
    _run_xgboost_lags, HAS_XGB,
)


def synthetic_revenue(seed: int = 42, days: int = 150) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    weekday = np.array([d.dayofweek for d in dates])
    trend = np.arange(days) * 0.4
    weekly = np.where(weekday < 5, 25.0, -12.0)
    noise = rng.normal(0, 7, days)
    return pd.Series(300 + trend + weekly + noise, index=dates, name="revenue")


def main(output_dir: str = "benchmark_results") -> dict:
    ts = synthetic_revenue()
    runners = {
        "seasonal_naive": _run_seasonal_naive,
        "gradient_boosting": _run_gradient_boosting,
        "random_forest": _run_random_forest,
    }
    if HAS_XGB:
        runners["xgboost_lags"] = _run_xgboost_lags
    rows = rolling_benchmark(ts, runners, horizon=7, folds=3, min_train=60)
    summary = aggregate_benchmark(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "fold_results.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "model_summary.csv", index=False)
    metadata = {
        "dataset": "synthetic; not evidence of real-world performance",
        "seed": 42, "days": len(ts), "horizon": 7, "folds": 3,
        "models": list(runners),
        "interpretation": "No winner certified; validate on representative revenue datasets.",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    return {"folds": rows, "summary": summary, "manifest": metadata}


if __name__ == "__main__":
    main()
