"""Reproducible chronological forecasting benchmark.

A standalone evaluation utility; does not change the production model selector.
All models receive identical train/test folds and forecast horizons.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class FoldScore:
    model: str
    fold: int
    train_size: int
    test_size: int
    mape: float | None
    wape: float | None
    rmse: float | None
    elapsed_seconds: float
    error: str | None = None


def score(actual, predicted) -> dict:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted values must be aligned finite nonempty arrays")
    diff = np.abs(y - p)
    positive = np.abs(y) > 1e-9
    denominator = np.abs(y).sum()
    return {
        "mape": float(np.mean(diff[positive] / np.abs(y[positive])) * 100) if positive.any() else None,
        "wape": float(diff.sum() / denominator * 100) if denominator > 1e-9 else None,
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
    }


def rolling_benchmark(series: pd.Series, runners: dict[str, Callable],
                      horizon: int = 7, folds: int = 3,
                      min_train: int = 30) -> list[dict]:
    """Expanding-window backtest; runners take (train_series, horizon).

    Train prefixes end strictly before test intervals. Exceptions are recorded,
    not treated as zero error. Timings are diagnostic, not stable hardware benchmarks.
    """
    if horizon < 1 or folds < 1 or min_train < 1:
        raise ValueError("horizon, folds and min_train must be positive")
    if not isinstance(series.index, pd.DatetimeIndex) or not series.index.is_unique:
        raise ValueError("Expected a unique DatetimeIndex")
    ts = series.sort_index().astype(float)
    if not np.isfinite(ts.to_numpy()).all():
        raise ValueError("Series must contain finite observations")
    if len(ts) < min_train + horizon * folds:
        raise ValueError("Insufficient observations for requested chronological folds")
    output = []
    for i in range(folds):
        start = len(ts) - (folds - i) * horizon
        train = ts.iloc[:start]
        test = ts.iloc[start:start + horizon]
        for name, runner in runners.items():
            begun = perf_counter()
            try:
                prediction = runner(train.copy(), len(test))
                metrics = score(test.to_numpy(), prediction)
                error = None
            except Exception as exc:
                metrics = {"mape": None, "wape": None, "rmse": None}
                error = f"{type(exc).__name__}: {exc}"
            output.append(asdict(FoldScore(
                model=name, fold=i + 1, train_size=len(train),
                test_size=len(test), elapsed_seconds=round(perf_counter() - begun, 6),
                error=error, **metrics)))
    return output


def aggregate_benchmark(rows: list[dict]) -> list[dict]:
    """Aggregate only fully successful models; never hide failed folds."""
    result = []
    for name in dict.fromkeys(row["model"] for row in rows):
        group = [row for row in rows if row["model"] == name]
        failures = [r for r in group if r["error"]]
        result.append({
            "model": name, "folds": len(group), "failed_folds": len(failures),
            "mape": float(np.mean([r["mape"] for r in group])) if not failures and all(r["mape"] is not None for r in group) else None,
            "wape": float(np.mean([r["wape"] for r in group])) if not failures and all(r["wape"] is not None for r in group) else None,
            "rmse": float(np.mean([r["rmse"] for r in group])) if not failures else None,
            "total_seconds": round(sum(r["elapsed_seconds"] for r in group), 6),
        })
    return result
