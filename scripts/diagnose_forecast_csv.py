"""Read-only forecast diagnostics for a user-supplied billing CSV.

Run locally:
  python -m scripts.diagnose_forecast_csv electricity_billing_v1.csv
No data uploaded to CI or GitHub; no production model logic is changed.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

from capabilities.forecasting import prepare_series, _run_seasonal_naive


def diagnose(path: str, date_col: str = "bill_date",
             target_col: str = "billed_amount",
             horizon: int = 30, folds: int = 3) -> dict:
    df = pd.read_csv(path, usecols=[date_col, target_col], parse_dates=[date_col])
    observed = df.groupby(date_col)[target_col].sum().sort_index()
    results = {"rows": len(df), "observed_days": len(observed),
               "calendar_days": int((observed.index.max()-observed.index.min()).days + 1),
               "days_below_100": int((observed < 100).sum()),
               "days_below_1000": int((observed < 1000).sum()),
               "total_billed": float(observed.sum()), "scenarios": {}}
    for label, prep in {
        "observed_dates_only": {},
        "daily_interpolated": {"fill_gaps": True},
        "daily_interpolated_winsorized": {"fill_gaps": True, "winsorize": True},
    }.items():
        ts = prepare_series(df, date_col, target_col, prep)
        fold_rows = []
        for i in range(folds):
            start = len(ts) - (folds-i)*horizon
            if start < 7:
                raise ValueError("Insufficient data for diagnostic folds")
            train, test = ts.iloc[:start], ts.iloc[start:start+horizon]
            pred = _run_seasonal_naive(train, len(test), period=7)
            actual = test.to_numpy(dtype=float)
            finite = np.isfinite(actual) & np.isfinite(pred)
            if not finite.all():
                raise ValueError("Nonfinite values in diagnostic fold")
            error = np.abs(actual-pred)
            nonzero = np.abs(actual)>1e-9
            fold_rows.append({
                "fold": i+1,
                "train_end": str(train.index[-1].date()),
                "test_start": str(test.index[0].date()),
                "test_end": str(test.index[-1].date()),
                "mape": float(np.mean(error[nonzero]/np.abs(actual[nonzero]))*100) if nonzero.any() else None,
                "wape": float(error.sum()/np.abs(actual).sum()*100) if np.abs(actual).sum()>1e-9 else None,
                "rmse": float(np.sqrt(np.mean((actual-pred)**2))),
                "absolute_error": float(error.sum()),
                "absolute_actual": float(np.abs(actual).sum()),
            })
        results["scenarios"][label] = {
            "folds": fold_rows,
            "mean_mape": float(np.mean([f["mape"] for f in fold_rows if f["mape"] is not None])),
            "pooled_wape": float(sum(f["absolute_error"] for f in fold_rows) /
                                 sum(f["absolute_actual"] for f in fold_rows)*100),
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--date-col", default="bill_date")
    parser.add_argument("--target-col", default="billed_amount")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.csv_path, args.date_col, args.target_col), indent=2))
