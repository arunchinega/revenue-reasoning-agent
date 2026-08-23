"""Generate synthetic utilities billing data with planted seasonality,
changepoint, outliers, and leakage rows — for pipeline tests + demo."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def make_utilities_csv(path: str = "demo_utilities.csv", days: int = 540,
                       n_customers: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    rows = []
    tariffs = {"residential": 7.5, "commercial": 9.2, "industrial": 6.8}

    for cust in range(n_customers):
        seg = rng.choice(list(tariffs), p=[0.55, 0.3, 0.15])
        rate = tariffs[seg]
        base = rng.uniform(20, 400)
        # 70% active full period; 30% churn out or join late (mild base drift)
        if rng.random() < 0.7:
            start, end = 0, days
        elif rng.random() < 0.5:
            start, end = 0, int(rng.integers(days // 2, days))          # churns
        else:
            start, end = int(rng.integers(0, days // 2)), days          # joins
        for d in dates[start:end]:
            d = pd.Timestamp(d)
            weekly = 1 + 0.25 * np.sin(2 * np.pi * d.dayofweek / 7)
            trend = 1 + 0.0004 * (d - dates[0]).days
            change = 1.15 if d > pd.Timestamp("2025-10-01") else 1.0  # changepoint
            kwh = max(base * weekly * trend * change * rng.normal(1, 0.12), 1)
            billed = kwh * rate
            rows.append({
                "bill_date": d.date(), "customer_id": f"CUST{cust:04d}",
                "segment": seg, "tariff_rate": rate,
                "kwh_consumed": round(kwh, 2), "billed_amount": round(billed, 2),
            })

    df = pd.DataFrame(rows).sort_values("bill_date").reset_index(drop=True)

    # --- planted defects ------------------------------------------------------
    idx = rng.choice(df.index, 12, replace=False)
    correct_before = df.loc[idx, "billed_amount"].copy()   # ground truth snapshot
    df.loc[idx[:4], "billed_amount"] *= 0.4      # under-billing (leakage)
    df.loc[idx[4:7], "billed_amount"] = 0.0      # zero-billed with usage (leakage)
    df.loc[idx[7:10], "tariff_rate"] *= 0.5      # tariff misapplication (leakage)
    df.loc[idx[7:10], "billed_amount"] = (
        df.loc[idx[7:10], "kwh_consumed"] * df.loc[idx[7:10], "tariff_rate"]
    ).round(2)
    df.loc[idx[10:], "kwh_consumed"] *= 8        # consumption spikes (anomaly)
    df.loc[idx[10:], "billed_amount"] = (
        df.loc[idx[10:], "kwh_consumed"] * df.loc[idx[10:], "tariff_rate"]
    ).round(2)
    # some missing values for realism
    miss = rng.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[miss, "kwh_consumed"] = np.nan

    # ground-truth manifest: planted leakage = correct - corrupted (rows 0..9)
    leak_idx = idx[:10]
    planted = (correct_before.loc[leak_idx] - df.loc[leak_idx, "billed_amount"]).clip(lower=0)
    manifest = {
        "planted_leakage_total": round(float(planted.sum()), 2),
        "by_type": {
            "under_billing": round(float(planted.loc[idx[:4]].sum()), 2),
            "zero_billed": round(float(planted.loc[idx[4:7]].sum()), 2),
            "tariff_misapplication": round(float(planted.loc[idx[7:10]].sum()), 2),
        },
        "defect_rows": {str(int(i)): round(float(planted.loc[i]), 2) for i in leak_idx},
    }
    import json
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))

    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = make_utilities_csv()
    print(f"wrote demo_utilities.csv: {df.shape}")
