"""
tests/make_external_data.py — the "as-specified" demo dataset.

Implements the exact spec we wrote for external generation (GPT/Gemini prompt):
60 customers, 18 months daily, string tariff_code, weekly seasonality + trend,
stable forecastable aggregate, and five planted defect families:

  D1 unbilled usage        (kwh normal, billed = 0)          → leakage
  D2 tariff misapplication (COM billed at RES rate)          → leakage
  D3 systematic under-billing (final 60 days × U(0.70,0.88)) → leakage
  D4 revenue spikes        (single days 8-15×)               → anomaly
  D5 zero-usage billing    (kwh = 0, billed normal)          → anomaly

Ground truth goes to <name>.manifest.json; run `python tests/make_external_data.py`.
Deterministic: seed 42.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

RATES = {"RES-STD": 6.5, "RES-TOU": 7.0, "COM-STD": 8.2, "IND-STD": 7.1}
FIXED = {"residential": 120.0, "commercial": 450.0, "industrial": 2000.0}
BASE_KWH = {"residential": 30.0, "commercial": 120.0, "industrial": 800.0}


def make_external_csv(path: str, days: int = 540, n_customers: int = 60,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize().replace(day=1) - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")

    customers = []
    for i in range(n_customers):
        seg = "residential" if i < int(n_customers * 0.70) else (
            "commercial" if i < int(n_customers * 0.95) else "industrial")
        tc = "COM-STD" if seg == "commercial" else (
            "IND-STD" if seg == "industrial" else
            ("RES-TOU" if rng.random() < 0.10 else "RES-STD"))
        r = rng.random()
        if r < 0.70:
            start, stop = 0, days                       # stable base
        elif r < 0.90:
            start, stop = int(rng.integers(30, days // 2)), days   # joiner
        else:
            start, stop = 0, int(rng.integers(days // 2, days - 30))  # churner
        customers.append((f"CUST{i + 1:04d}", seg, tc, start, stop))

    rows = []
    for cid, seg, tc, start, stop in customers:
        base = BASE_KWH[seg]
        for j, d in enumerate(dates):
            if not (start <= j < stop):
                continue
            weekend = d.dayofweek >= 5
            season = (0.80 if weekend else 1.0) if seg == "commercial" else \
                     (1.10 if weekend else 1.0) if seg == "residential" else 1.0
            trend = 1 + 0.08 * j / 365.0
            kwh = max(base * season * trend * rng.normal(1, 0.06), 0.5)
            billed = kwh * RATES[tc] + FIXED[seg] / 30.0
            rows.append([cid, d, seg, tc, round(kwh, 2), round(billed, 2)])

    df = pd.DataFrame(rows, columns=["customer_id", "bill_date", "segment",
                                     "tariff_code", "kwh_consumed", "billed_amount"])
    df = df.sort_values(["bill_date", "customer_id"]).reset_index(drop=True)

    # ---------------- planted defects (non-overlapping customer sets) --------
    manifest = {"defects": [], "summary": {}}
    active = [c for c in customers if c[3] == 0 and c[4] == len(dates)]  # stable only
    picks = rng.choice(len(active), 14, replace=False)
    stable_ids = [active[k][0] for k in picks]
    d1_ids, d2_pool, d3_ids, d5_ids = (stable_ids[:4], stable_ids[4:9],
                                       stable_ids[9:12], stable_ids[12:14])
    # D2 needs commercial customers specifically
    com_stable = [c[0] for c in active if c[1] == "commercial"
                  and c[0] not in d1_ids + d3_ids + d5_ids]
    d2_ids = com_stable[:3]

    def _plant(cids, dtype, fn, win_lo, win_hi):
        for cid in cids:
            span = int(rng.integers(win_lo, win_hi + 1))
            lo = int(rng.integers(60, days - span - 30))
            w0, w1 = dates[lo], dates[lo + span - 1]
            m = (df["customer_id"] == cid) & df["bill_date"].between(w0, w1)
            impact = fn(m)
            manifest["defects"].append({
                "defect_type": dtype, "customer_id": cid,
                "start_date": str(w0.date()), "end_date": str(w1.date()),
                "expected_revenue_impact": round(float(impact), 2)})

    def _unbilled(m):
        imp = df.loc[m, "billed_amount"].sum()
        df.loc[m, "billed_amount"] = 0.0
        return imp

    def _tariff_mis(m):
        correct = df.loc[m, "billed_amount"]
        wrong = df.loc[m, "kwh_consumed"] * RATES["RES-STD"] + FIXED["commercial"] / 30.0
        df.loc[m, "billed_amount"] = wrong.round(2)
        return (correct - wrong).clip(lower=0).sum()

    def _under(m):
        u = rng.uniform(0.70, 0.88)
        correct = df.loc[m, "billed_amount"]
        df.loc[m, "billed_amount"] = (correct * u).round(2)
        return (correct * (1 - u)).sum()

    def _zero_usage(m):
        df.loc[m, "kwh_consumed"] = 0.0
        return 0.0  # anomaly, not leakage impact

    _plant(d1_ids, "D1_unbilled_usage", _unbilled, 10, 20)
    _plant(d2_ids, "D2_tariff_misapplication", _tariff_mis, 30, 45)
    # D3: entire final 60 days
    for cid in d3_ids:
        m = (df["customer_id"] == cid) & (df["bill_date"] > dates[-60])
        u = rng.uniform(0.70, 0.88)
        correct = df.loc[m, "billed_amount"]
        df.loc[m, "billed_amount"] = (correct * u).round(2)
        manifest["defects"].append({
            "defect_type": "D3_under_billing", "customer_id": cid,
            "start_date": str(dates[-59].date()), "end_date": str(dates[-1].date()),
            "expected_revenue_impact": round(float((correct * (1 - u)).sum()), 2)})
    # D4: 6 single-day spikes on random stable customers (anomaly only)
    spike_ids = rng.choice(len(active), 6, replace=False)
    for k in spike_ids:
        cid = active[k][0]
        sub = df.index[(df["customer_id"] == cid)]
        i = int(rng.choice(sub))
        mult = float(rng.uniform(8, 15))
        df.loc[i, "billed_amount"] = round(df.loc[i, "billed_amount"] * mult, 2)
        manifest["defects"].append({
            "defect_type": "D4_revenue_spike", "customer_id": cid,
            "start_date": str(df.loc[i, "bill_date"].date()),
            "end_date": str(df.loc[i, "bill_date"].date()),
            "expected_revenue_impact": 0.0})
    _plant(d5_ids, "D5_zero_usage_billing", _zero_usage, 5, 5)

    # ---------------- normal data mess ---------------------------------------
    miss = rng.choice(df.index, int(len(df) * 0.01), replace=False)
    df.loc[miss, "kwh_consumed"] = np.nan
    dupes = df.sample(6, random_state=seed)
    df = pd.concat([df, dupes]).sort_values(["bill_date", "customer_id"]).reset_index(drop=True)

    # ---------------- manifest summary ---------------------------------------
    by_type: dict[str, float] = {}
    for d in manifest["defects"]:
        by_type[d["defect_type"]] = round(
            by_type.get(d["defect_type"], 0.0) + d["expected_revenue_impact"], 2)
    manifest["summary"] = {
        "total_leakage_impact": round(sum(v for k, v in by_type.items()
                                          if k.startswith(("D1", "D2", "D3"))), 2),
        "by_type": by_type,
        "rows": len(df), "customers": n_customers,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
    }

    df.to_csv(path, index=False)
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return df


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(
        Path(tempfile.gettempdir()) / "utilities_billing.csv")
    df = make_external_csv(out)
    man = json.loads(Path(out).with_suffix(".manifest.json").read_text())
    print(f"rows: {man['summary']['rows']} | customers: {man['summary']['customers']} "
          f"| range: {man['summary']['date_range']}")
    print("planted leakage by type:", man["summary"]["by_type"])
    print("TOTAL leakage impact:", f"{man['summary']['total_leakage_impact']:,.0f}")
    print("saved:", out, "+ manifest")
