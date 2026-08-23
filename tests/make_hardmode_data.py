"""Hard-mode synthetic data: extreme billing-system glitch spikes pollute the
series enough that first-pass forecasts fail acceptance, but winsorization
recovers them — designed to force the Critic's retry path."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_hardmode_csv(path: str = "hardmode_utilities.csv", days: int = 400,
                      n_customers: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    tariffs = {"residential": 7.5, "commercial": 9.2}
    rows = []
    for cust in range(n_customers):
        seg = rng.choice(list(tariffs), p=[0.6, 0.4])
        rate = tariffs[seg]
        base = rng.uniform(30, 300)
        for d in dates:
            d = pd.Timestamp(d)
            weekly = 1 + 0.3 * np.sin(2 * np.pi * d.dayofweek / 7)
            trend = 1 + 0.0008 * (d - dates[0]).days
            kwh = max(base * weekly * trend * rng.normal(1, 0.25), 1)  # noisier than demo
            rows.append({
                "bill_date": d.date(), "customer_id": f"CUST{cust:04d}",
                "segment": seg, "tariff_rate": rate,
                "kwh_consumed": round(kwh, 2),
                "billed_amount": round(kwh * rate, 2),
            })
    df = pd.DataFrame(rows)

    # --- pollution: billing-system glitch days with extreme duplicate-magnitude
    # spikes concentrated in the BACKTEST region (last ~120 days) so first-pass
    # fits fail where it counts -------------------------------------------------
    glitch_days = rng.choice(dates[-150:], size=4, replace=False)
    glitch_mask = df["bill_date"].isin(pd.Series(glitch_days).dt.date)
    idx = df.index[glitch_mask]
    df.loc[idx, "billed_amount"] = (df.loc[idx, "billed_amount"] * rng.uniform(
        25, 60, size=len(idx))).round(2)

    # sprinkle row-level moderate outliers so EDA flags outlier-heavy target
    extra = rng.choice(df.index[~glitch_mask], size=int(len(df) * 0.02), replace=False)
    df.loc[extra, "billed_amount"] = (df.loc[extra, "billed_amount"]
                                      * rng.uniform(4, 8, size=len(extra))).round(2)

    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    d = make_hardmode_csv()
    print("wrote", d.shape)
