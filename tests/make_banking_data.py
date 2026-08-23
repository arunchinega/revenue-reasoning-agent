"""
tests/make_banking_data.py — third vertical: banking fee income.

120 accounts, 540 days of transactions across 3 products with a fee schedule
(flat + %), month-end salary surges + weekly cycle + growth trend — structure
that rewards ML models (calendar + nonlinear covariate effects) in the
bake-off. Planted defects + manifest:

  B1 fee waived with activity   (fee=0 while txn>0, windows)   → leakage
  B2 fee schedule mismatch      (wrong product's rate applied) → leakage
  B3 systematic under-collection (final 40 days × U(0.7,0.85)) → leakage
  B4 fee spikes                 (single txns 8-12×)            → anomaly
  B5 duplicate fee rows                                        → anomaly/leak

Deterministic: seed 13. Run: python tests/make_banking_data.py [out.csv]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = {"savings": (15.0, 0.0008), "current": (45.0, 0.0015),
            "loan_servicing": (120.0, 0.0025)}   # (flat_fee, pct_of_txn)
REGIONS = ["Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Pune"]


def make_banking_csv(path: str, days: int = 540, n_accounts: int = 120,
                     seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    accounts = []
    for i in range(n_accounts):
        prod = rng.choice(list(PRODUCTS), p=[0.5, 0.35, 0.15])
        region = rng.choice(REGIONS)
        rate = rng.uniform(0.2, 0.8)
        accounts.append((f"ACC{i + 1:04d}", prod, region, rate))

    rows = []
    for aid, prod, region, rate in accounts:
        flat, pct = PRODUCTS[prod]
        for j, d in enumerate(dates):
            month_end = d.day >= 26
            p_txn = rate * (1.8 if month_end else 1.0) * (0.75 if d.dayofweek >= 5 else 1.0)
            if rng.random() > min(p_txn, 0.97):
                continue
            trend = 1 + 0.12 * j / 365.0
            txn = float(rng.lognormal(mean=9.2 if prod == "loan_servicing" else 8.2,
                                      sigma=0.7)) * trend * (1.5 if month_end else 1.0)
            fee = flat + pct * txn
            rows.append([f"TXN{len(rows) + 1:06d}", aid, d, prod, region,
                         round(txn, 2), round(fee, 2)])

    df = pd.DataFrame(rows, columns=["txn_id", "account_id", "txn_date",
                                     "product", "region", "txn_amount",
                                     "fee_amount"])
    df = df.sort_values(["txn_date", "account_id"]).reset_index(drop=True)

    manifest = {"defects": [], "summary": {}}

    def _log(dtype, aid, w0, w1, impact):
        manifest["defects"].append({
            "defect_type": dtype, "customer_id": aid,
            "start_date": str(pd.Timestamp(w0).date()) if w0 is not None else None,
            "end_date": str(pd.Timestamp(w1).date()) if w1 is not None else None,
            "expected_revenue_impact": round(float(impact), 2)})

    ids = [a[0] for a in accounts]
    non_sav = [a[0] for a in accounts if a[1] != "savings"]
    sav_ok = [a[0] for a in accounts]
    picks = (list(rng.choice(sav_ok, 6, replace=False))
             + list(rng.choice(non_sav, 6, replace=False))
             + list(rng.choice(sav_ok, 4, replace=False)))
    picks = list(dict.fromkeys(picks))[:16]
    while len(picks) < 16:
        extra = str(rng.choice(sav_ok))
        if extra not in picks:
            picks.append(extra)
    b1, b2, b3 = picks[:6], picks[6:12], picks[12:16]

    for aid in b1:                                  # B1 waived fees
        span = int(rng.integers(20, 35))
        lo = int(rng.integers(60, days - span - 70))
        w0, w1 = dates[lo], dates[lo + span - 1]
        m = (df["account_id"] == aid) & df["txn_date"].between(w0, w1)
        _log("B1_fee_waived", aid, w0, w1, df.loc[m, "fee_amount"].sum())
        df.loc[m, "fee_amount"] = 0.0

    for aid in b2:                                  # B2 wrong (cheaper) schedule
        acct_prod = next(a[1] for a in accounts if a[0] == aid)
        order = ["savings", "current", "loan_servicing"]
        cheaper = order[max(order.index(acct_prod) - 1, 0)]
        if cheaper == acct_prod:
            continue
        wf, wp = PRODUCTS[cheaper]
        span = int(rng.integers(45, 75))
        lo = int(rng.integers(60, days - span - 50))
        w0, w1 = dates[lo], dates[lo + span - 1]
        m = (df["account_id"] == aid) & df["txn_date"].between(w0, w1)
        correct = df.loc[m, "fee_amount"]
        new_fee = wf + wp * df.loc[m, "txn_amount"]
        _log("B2_schedule_mismatch", aid, w0, w1, (correct - new_fee).sum())
        df.loc[m, "fee_amount"] = new_fee.round(2)

    for aid in b3:                                  # B3 under-collection
        m = (df["account_id"] == aid) & (df["txn_date"] > dates[-60])
        u = rng.uniform(0.55, 0.75)
        correct = df.loc[m, "fee_amount"]
        _log("B3_under_collection", aid, dates[-39], dates[-1],
             (correct * (1 - u)).sum())
        df.loc[m, "fee_amount"] = (correct * u).round(2)

    for aid in list(rng.choice(ids, 6, replace=False)):   # B4 spikes
        idxs = df.index[df["account_id"] == aid]
        i = int(rng.choice(idxs))
        df.loc[i, "fee_amount"] = round(df.loc[i, "fee_amount"] *
                                        float(rng.uniform(8, 12)), 2)
        _log("B4_fee_spike", aid, df.loc[i, "txn_date"], df.loc[i, "txn_date"], 0.0)

    dup = df.sample(8, random_state=seed)                 # B5 duplicates
    df = pd.concat([df, dup]).sort_values(["txn_date", "account_id"]).reset_index(drop=True)
    _log("B5_duplicate_fees", "various", None, None, dup["fee_amount"].sum())

    miss = np.random.default_rng(seed + 1).choice(df.index, int(len(df) * 0.006),
                                                  replace=False)
    df.loc[miss, "txn_amount"] = np.nan

    by_type: dict[str, float] = {}
    for d0 in manifest["defects"]:
        by_type[d0["defect_type"]] = round(
            by_type.get(d0["defect_type"], 0.0) + d0["expected_revenue_impact"], 2)
    manifest["summary"] = {
        "total_leakage_impact": round(sum(v for k, v in by_type.items()
                                          if k.startswith(("B1", "B2", "B3"))), 2),
        "by_type": by_type, "rows": len(df), "accounts": n_accounts,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
    }
    df.to_csv(path, index=False)
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(tempfile.gettempdir()) / "banking_fees.csv")
    make_banking_csv(out)
    man = json.loads(Path(out).with_suffix(".manifest.json").read_text())
    print(f"rows: {man['summary']['rows']} | accounts: {man['summary']['accounts']}")
    print("planted:", man["summary"]["by_type"])
    print("TOTAL leakage:", f"{man['summary']['total_leakage_impact']:,.0f}")
    print("saved:", out, "+ manifest")
