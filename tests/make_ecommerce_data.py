"""
tests/make_ecommerce_data.py — second demo dataset: Indian e-commerce.

5 states (Maharashtra, Karnataka, Telangana, Tamil Nadu, Delhi), 18 months of
daily orders, 80 customers, 4 categories. Planted defects with a ground-truth
manifest, ecommerce-flavoured:

  E1 zero-charged orders     (units shipped, amount = 0)         → leakage
  E2 discount abuse          (discount stuck at 60-80% a window) → leakage
  E3 systematic under-charge (final 45 days × U(0.72,0.86))      → leakage
  E4 price-error spikes      (single days 8-14×)                 → anomaly
  E5 duplicate orders        (same order twice)                  → anomaly/leakage

Also a REGIONAL DIP: Telangana revenue drops 25% for the final 6 weeks —
the RCA capability's story beat ("why did revenue move? → concentrated in
state=Telangana").

Deterministic: seed 7. Run: python tests/make_ecommerce_data.py [out.csv]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

STATES = ["Maharashtra", "Karnataka", "Telangana", "Tamil Nadu", "Delhi"]
STATE_W = [0.28, 0.22, 0.18, 0.18, 0.14]
CATS = {"electronics": 5200.0, "fashion": 1400.0, "grocery": 650.0, "home": 2100.0}


def make_ecommerce_csv(path: str, days: int = 540, n_customers: int = 80,
                       seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    customers = []
    for i in range(n_customers):
        state = rng.choice(STATES, p=STATE_W)
        fav = rng.choice(list(CATS))
        rate = rng.uniform(0.25, 0.9)          # orders per day propensity
        r = rng.random()
        if r < 0.75:
            start, stop = 0, days
        elif r < 0.92:
            start, stop = int(rng.integers(30, days // 2)), days
        else:
            start, stop = 0, int(rng.integers(days // 2, days - 30))
        customers.append((f"CUST{i + 1:04d}", state, fav, rate, start, stop))

    rows = []
    for cid, state, fav, rate, start, stop in customers:
        for j, d in enumerate(dates):
            if not (start <= j < stop) or rng.random() > rate:
                continue
            weekend = d.dayofweek >= 5
            season = 1.25 if weekend else 1.0                    # weekend bump
            trend = 1 + 0.10 * j / 365.0
            cat = fav if rng.random() < 0.6 else rng.choice(list(CATS))
            units = int(rng.integers(1, 4 if cat != "grocery" else 9))
            price = CATS[cat] * rng.normal(1, 0.08) * trend * season
            disc = float(rng.choice([0, 0, 0, 5, 10, 15], p=[.45, .15, .1, .12, .1, .08]))
            amount = units * price * (1 - disc / 100)
            rows.append([f"ORD{len(rows) + 1:06d}", cid, d, state, cat, units,
                         round(price, 2), disc, round(amount, 2)])

    df = pd.DataFrame(rows, columns=["order_id", "customer_id", "order_date",
                                     "state", "category", "units", "unit_price",
                                     "discount_pct", "order_amount"])
    df = df.sort_values(["order_date", "customer_id"]).reset_index(drop=True)

    manifest = {"defects": [], "summary": {}}
    stable = [c for c in customers if c[4] == 0 and c[5] == days]
    picks = rng.choice(len(stable), 15, replace=False)
    ids = [stable[k][0] for k in picks]
    e1_ids, e2_ids, e3_ids = ids[:4], ids[4:8], ids[8:11]

    def _log(dtype, cid, w0, w1, impact):
        manifest["defects"].append({
            "defect_type": dtype, "customer_id": cid,
            "start_date": str(w0.date()) if w0 is not None else None,
            "end_date": str(w1.date()) if w1 is not None else None,
            "expected_revenue_impact": round(float(impact), 2)})

    # E1 zero-charged orders
    for cid in e1_ids:
        span = int(rng.integers(8, 16))
        lo = int(rng.integers(60, days - span - 40))
        w0, w1 = dates[lo], dates[lo + span - 1]
        m = (df["customer_id"] == cid) & df["order_date"].between(w0, w1)
        _log("E1_zero_charged", cid, w0, w1, df.loc[m, "order_amount"].sum())
        df.loc[m, "order_amount"] = 0.0

    # E2 discount abuse (60-80% stuck)
    for cid in e2_ids:
        span = int(rng.integers(20, 35))
        lo = int(rng.integers(60, days - span - 40))
        w0, w1 = dates[lo], dates[lo + span - 1]
        m = (df["customer_id"] == cid) & df["order_date"].between(w0, w1)
        correct = df.loc[m, "order_amount"]
        bad_disc = rng.uniform(60, 80)
        base = correct / (1 - df.loc[m, "discount_pct"] / 100)
        new_amt = base * (1 - bad_disc / 100)
        _log("E2_discount_abuse", cid, w0, w1, (correct - new_amt).sum())
        df.loc[m, "discount_pct"] = round(bad_disc, 1)
        df.loc[m, "order_amount"] = new_amt.round(2)

    # E3 systematic under-charge, final 45 days
    for cid in e3_ids:
        m = (df["customer_id"] == cid) & (df["order_date"] > dates[-45])
        u = rng.uniform(0.72, 0.86)
        correct = df.loc[m, "order_amount"]
        _log("E3_under_charge", cid, str(dates[-44].date()) and dates[-44],
             dates[-1], (correct * (1 - u)).sum())
        df.loc[m, "order_amount"] = (correct * u).round(2)

    # E4 price-error spikes (anomaly, impact 0)
    for k in rng.choice(len(stable), 7, replace=False):
        cid = stable[k][0]
        idxs = df.index[df["customer_id"] == cid]
        i = int(rng.choice(idxs))
        df.loc[i, "order_amount"] = round(df.loc[i, "order_amount"] *
                                          float(rng.uniform(8, 14)), 2)
        _log("E4_price_spike", cid, df.loc[i, "order_date"],
             df.loc[i, "order_date"], 0.0)

    # E5 duplicate orders
    dup = df.sample(10, random_state=seed)
    df = pd.concat([df, dup]).sort_values(["order_date", "customer_id"]).reset_index(drop=True)
    _log("E5_duplicate_orders", "various", None, None,
         dup["order_amount"].sum())

    # REGIONAL DIP: Telangana loses ~25% of ORDERS for final 6 weeks —
    # demand softness (RCA beat), implemented as missing orders so it is
    # correctly NOT leakage (customers who didn't buy owe nothing)
    m = (df["state"] == "Telangana") & (df["order_date"] > dates[-42])
    drop_idx = df.index[m]
    drop_idx = np.random.default_rng(seed + 2).choice(
        drop_idx, int(len(drop_idx) * 0.25), replace=False)
    df = df.drop(index=drop_idx).reset_index(drop=True)
    manifest["regional_dip"] = {"state": "Telangana", "window_days": 42,
                                "drop_pct": 25}

    # mess
    miss = np.random.default_rng(seed + 1).choice(df.index, int(len(df) * 0.008),
                                                  replace=False)
    df.loc[miss, "units"] = np.nan

    by_type: dict[str, float] = {}
    for d in manifest["defects"]:
        by_type[d["defect_type"]] = round(
            by_type.get(d["defect_type"], 0.0) + d["expected_revenue_impact"], 2)
    manifest["summary"] = {
        "total_leakage_impact": round(sum(v for k, v in by_type.items()
                                          if k.startswith(("E1", "E2", "E3"))), 2),
        "by_type": by_type, "rows": len(df),
        "customers": n_customers, "states": STATES,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
    }
    df.to_csv(path, index=False)
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(tempfile.gettempdir()) / "ecommerce_india.csv")
    make_ecommerce_csv(out)
    man = json.loads(Path(out).with_suffix(".manifest.json").read_text())
    print(f"rows: {man['summary']['rows']} | customers: {man['summary']['customers']}"
          f" | states: {len(man['summary']['states'])}")
    print("planted leakage:", man["summary"]["by_type"])
    print("TOTAL leakage:", f"{man['summary']['total_leakage_impact']:,.0f}",
          "| regional dip: Telangana -25% final 6 weeks")
    print("saved:", out, "+ manifest")
