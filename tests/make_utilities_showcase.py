"""
tests/make_utilities_showcase.py — demo-grade synthetic utilities billing data.

Built so that EVERY capability in the agent has something real to find:

  forecasting   weekly + yearly (summer AC peak) seasonality, gentle growth
                trend, and a demand changepoint (new industrial park goes live)
  anomaly       consumption spikes, meter read errors (near-zero day then a
                catch-up day), stuck meters
  leakage       under-billing, zero-billed with usage, tariff misapplication
                (rate below the modal rate of its connection type), duplicate
                billing rows
  rca           a 30-day feeder outage in one region (revenue drops ~55% there)
  segmentation  five customer archetypes with distinct RFM profiles plus
                churners and late joiners
  what-if /     derived from the above
  recommend

Column names deliberately hit the utilities domain signals
(meter, kwh, tariff, bill_date, feeder, connection) and the column-mapper
hints (customer_id → id, billed_amount → target, tariff_rate → price,
kwh_consumed → quantity).

Writes <path>.csv and <path>.manifest.json (ground truth for reconciliation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

START = "2025-01-01"
TARIFFS = {            # INR / kWh, constant per connection type (no rate changepoint:
    "residential": 7.50,   # a rate revision would make every pre-revision row look
    "commercial": 9.20,    # like a tariff misapplication)
    "industrial": 6.80,
    "agricultural": 4.10,
    "government": 8.40,
}
REGIONS = ["North", "South", "East", "West", "Central"]
RCA_REGION = "North"


def make_utilities_showcase(path: str = "demo_utilities_showcase.csv",
                            days: int = 548, n_customers: int = 110,
                            seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days, freq="D")
    t0 = dates[0]
    change_day = pd.Timestamp("2025-10-01")              # demand changepoint
    rca_start = pd.Timestamp("2026-03-03")               # feeder outage window
    rca_end = rca_start + pd.Timedelta(days=29)

    # ---- customer archetypes (segmentation ground truth) --------------------
    archetypes = [
        # name, connection, base kWh/day range, weight
        ("small_home", "residential", (6, 18), 0.38),
        ("large_home", "residential", (25, 60), 0.17),
        ("shop", "commercial", (60, 180), 0.20),
        ("factory", "industrial", (600, 1800), 0.10),
        ("farm", "agricultural", (80, 300), 0.09),
        ("office", "government", (150, 400), 0.06),
    ]
    weights = np.array([a[3] for a in archetypes]); weights /= weights.sum()

    customers = []
    for i in range(n_customers):
        a = archetypes[rng.choice(len(archetypes), p=weights)]
        region = rng.choice(REGIONS)
        feeder = f"{region[:1]}-F{rng.integers(1, 4):02d}"
        # lifecycle: 72% full period, 14% churn, 14% join late
        r = rng.random()
        if r < 0.72:
            s, e = 0, days
        elif r < 0.86:
            s, e = 0, int(rng.integers(days // 2, days - 30))
        else:
            s, e = int(rng.integers(30, days // 2)), days
        customers.append(dict(
            customer_id=f"CUST{i:04d}", meter_id=f"MTR{rng.integers(100000, 999999)}",
            archetype=a[0], connection_type=a[1], region=region, feeder=feeder,
            base=float(rng.uniform(*a[2])), start=s, end=e,
            noise=0.08 if a[1] in ("industrial", "government") else 0.16,
        ))

    # ---- daily rows ---------------------------------------------------------
    doy = np.array([d.dayofyear for d in dates])
    dow = np.array([d.dayofweek for d in dates])
    elapsed = np.arange(days)
    yearly = 1 + 0.22 * np.exp(-((doy - 150) / 45) ** 2)          # May–June AC peak
    yearly += 0.06 * np.exp(-((doy - 10) / 20) ** 2)               # small winter bump
    trend = 1 + 0.00035 * elapsed                                  # ~13%/yr growth
    change = np.where(dates > change_day, 1.0, 1.0)

    frames = []
    for c in customers:
        idx = np.arange(c["start"], c["end"])
        if c["connection_type"] in ("residential",):
            weekly = 1 + 0.18 * (dow[idx] >= 5)                    # homes: weekend up
        elif c["connection_type"] in ("commercial", "government", "industrial"):
            weekly = 1 - 0.30 * (dow[idx] == 6) - 0.10 * (dow[idx] == 5)  # Sunday off
        else:
            weekly = np.ones(len(idx))
        seas = yearly[idx] if c["connection_type"] != "industrial" else 1 + 0.4 * (yearly[idx] - 1)
        cp = np.ones(len(idx))
        if c["connection_type"] == "industrial":
            cp = np.where(dates[idx] > change_day, 1.35, 1.0)     # new industrial park
        kwh = c["base"] * weekly * seas * trend[idx] * cp * rng.normal(1, c["noise"], len(idx))
        kwh = np.clip(kwh, 0.5, None)
        rate = TARIFFS[c["connection_type"]]
        frames.append(pd.DataFrame({
            "bill_date": dates[idx].date,
            "customer_id": c["customer_id"], "meter_id": c["meter_id"],
            "connection_type": c["connection_type"], "region": c["region"],
            "feeder": c["feeder"], "tariff_rate": rate,
            "kwh_consumed": np.round(kwh, 2),
            "_arch": c["archetype"],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["bill_date"] = pd.to_datetime(df["bill_date"])

    # ---- RCA: feeder outage in one region ----------------------------------
    rca_mask = (df["region"] == RCA_REGION) & df["bill_date"].between(rca_start, rca_end)
    df.loc[rca_mask, "kwh_consumed"] = (df.loc[rca_mask, "kwh_consumed"] * 0.45).round(2)

    df["billed_amount"] = (df["kwh_consumed"] * df["tariff_rate"]).round(2)
    df = df.sort_values(["bill_date", "customer_id"]).reset_index(drop=True)

    manifest = {"defects": [], "summary": {}}

    def plant(kind, i, impact=0.0, note=""):
        manifest["defects"].append({
            "type": kind, "row_index": int(i),
            "customer_id": df.at[i, "customer_id"],
            "bill_date": str(df.at[i, "bill_date"].date()),
            "impact": round(float(impact), 2), "note": note})

    # eligible rows: active customers, outside the RCA window, decent usage
    pool = df.index[(~rca_mask.reindex(df.index, fill_value=False))
                    & (df["kwh_consumed"] > 5)].to_numpy().copy()
    rng.shuffle(pool)
    cursor = 0

    def take(n):
        nonlocal cursor
        out = pool[cursor:cursor + n]; cursor += n
        return out

    # ---- leakage: under-billing (billed well below kWh x tariff) ------------
    for i in take(22):
        correct = df.at[i, "billed_amount"]
        factor = rng.uniform(0.45, 0.72)
        df.at[i, "billed_amount"] = round(correct * factor, 2)
        plant("under_billing", i, correct - df.at[i, "billed_amount"],
              f"billed {factor:.0%} of expected")

    # ---- leakage: zero-billed despite usage ----------------------------------
    for i in take(16):
        correct = df.at[i, "billed_amount"]
        df.at[i, "billed_amount"] = 0.0
        plant("zero_billed_with_usage", i, correct, "no charge raised")

    # ---- leakage: tariff misapplication (lower band applied) -----------------
    for i in take(18):
        ct = df.at[i, "connection_type"]
        correct_rate = TARIFFS[ct]
        wrong = correct_rate * rng.choice([0.55, 0.62, 0.70, 0.80])
        correct = df.at[i, "billed_amount"]
        df.at[i, "tariff_rate"] = round(wrong, 2)
        df.at[i, "billed_amount"] = round(df.at[i, "kwh_consumed"] * wrong, 2)
        plant("tariff_misapplication", i, correct - df.at[i, "billed_amount"],
              f"{ct} billed at {wrong:.2f} instead of {correct_rate:.2f}")

    # ---- anomaly: consumption spikes (on big meters, so they stand out
    #      against the whole population, not just the customer's own history) --
    big = df.index[df["connection_type"].isin(["commercial", "industrial", "government"])
                   & (~rca_mask.reindex(df.index, fill_value=False))
                   & (df["kwh_consumed"] > 50)].to_numpy().copy()
    big = np.setdiff1d(big, pool[:cursor])       # never reuse a leakage row
    rng.shuffle(big)
    bcur = 0

    def take_big(n):
        nonlocal bcur
        out = big[bcur:bcur + n]; bcur += n
        return out

    for i in take_big(14):
        mult = rng.uniform(9.0, 14.0)
        df.at[i, "kwh_consumed"] = round(df.at[i, "kwh_consumed"] * mult, 2)
        df.at[i, "billed_amount"] = round(df.at[i, "kwh_consumed"] * df.at[i, "tariff_rate"], 2)
        plant("consumption_spike", i, 0.0, f"x{mult:.1f} usage")

    # ---- anomaly: stuck meter — three zero-read days, then one catch-up
    #      reading carrying all four days' consumption (~4x normal) -----------
    ind = [i for i in big[bcur:] if df.at[i, "connection_type"] == "industrial"][:8]
    bcur = len(big)                      # retire the big pool after this
    for i in ind:
        cid, d0 = df.at[i, "customer_id"], df.at[i, "bill_date"]
        run = df.index[(df["customer_id"] == cid)
                       & df["bill_date"].between(d0, d0 + pd.Timedelta(days=3))]
        if len(run) < 4:
            continue
        run = [int(r) for r in run]
        total = float(df.loc[run, "kwh_consumed"].sum())
        for r in run[:3]:
            df.at[r, "kwh_consumed"] = 0.0
            df.at[r, "billed_amount"] = 0.0
            plant("stuck_meter_zero_read", r, 0.0, "meter stuck, zero read (context row; a zero is not a population outlier, so not expected to flag)")
        j = run[3]
        df.at[j, "kwh_consumed"] = round(total, 2)
        df.at[j, "billed_amount"] = round(total * df.at[j, "tariff_rate"], 2)
        plant("stuck_meter_catchup", j, 0.0, "catch-up reading for 4 days")

    # ---- leakage: duplicate billing rows (same customer, date, amount) -------
    dup_src = np.setdiff1d(take(40), big)[:12]
    dups = df.loc[dup_src].copy()
    for i in dup_src:
        plant("duplicate_billing", i, df.at[i, "billed_amount"], "row billed twice")
    df = pd.concat([df, dups], ignore_index=True)
    # fix manifest indices for duplicates: the DUPLICATE (second) copy is the leak
    n_before = len(df) - len(dups)
    for n, d in enumerate([d for d in manifest["defects"] if d["type"] == "duplicate_billing"]):
        d["row_index"] = int(n_before + n)
        d["note"] = "second copy of a (customer, date, amount) row"

    # ---- realism: 0.8% missing kWh readings ----------------------------------
    miss = rng.choice(df.index, int(len(df) * 0.008), replace=False)
    df.loc[miss, "kwh_consumed"] = np.nan

    # ---- final order + output -----------------------------------------------
    df = df.sort_values(["bill_date", "customer_id"], kind="stable").reset_index(drop=True)
    # re-locate manifest rows after the sort by (customer, date, amount) key
    key = df.set_index(["customer_id", "bill_date", "billed_amount"]).index
    for d in manifest["defects"]:
        k = (d["customer_id"], pd.Timestamp(d["bill_date"]), None)
        hits = df.index[(df["customer_id"] == k[0]) & (df["bill_date"] == k[1])]
        d["row_index"] = [int(h) for h in hits]      # 1 row, or 2 for duplicates

    arch = df.pop("_arch")
    df["bill_date"] = df["bill_date"].dt.date
    cols = ["bill_date", "customer_id", "meter_id", "connection_type", "region",
            "feeder", "tariff_rate", "kwh_consumed", "billed_amount"]
    df = df[cols]

    by_type = {}
    for d in manifest["defects"]:
        s = by_type.setdefault(d["type"], {"rows": 0, "impact": 0.0})
        s["rows"] += 1; s["impact"] = round(s["impact"] + d["impact"], 2)
    leak_types = ("under_billing", "zero_billed_with_usage",
                  "tariff_misapplication", "duplicate_billing")
    rca_total = float(df.loc[(df["region"] == RCA_REGION)
                             & pd.to_datetime(df["bill_date"]).between(rca_start, rca_end),
                             "billed_amount"].sum())
    manifest["summary"] = {
        "rows": int(len(df)), "customers": n_customers, "days": days,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
        "planted_leakage_total": round(sum(by_type[t]["impact"] for t in leak_types
                                           if t in by_type), 2),
        "by_type": by_type,
        "forecasting": {
            "weekly_seasonality": "residential weekend +18%; commercial/industrial Sunday -30%",
            "yearly_seasonality": "May–June AC peak +22%, small January bump",
            "trend": "~13% per year",
            "changepoint": {"date": str(change_day.date()),
                            "effect": "industrial load +35% (new industrial park)"},
        },
        "rca": {"region": RCA_REGION, "window": [str(rca_start.date()), str(rca_end.date())],
                "effect": "feeder outage: consumption -55% in region",
                "revenue_in_window_after_outage": round(rca_total, 2),
                "approx_revenue_lost": round(rca_total / 0.45 - rca_total, 2)},
        "segmentation": {"archetypes": arch.value_counts().to_dict(),
                         "lifecycle": "72% full-period, 14% churn, 14% late joiners"},
        "missing_kwh_rows": int(len(miss)),
    }
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_utilities_showcase.csv"
    d = make_utilities_showcase(out)
    man = json.loads(Path(out).with_suffix(".manifest.json").read_text())
    print(f"wrote {out}: {d.shape}")
    print("planted leakage total:", man["summary"]["planted_leakage_total"])
    for k, v in man["summary"]["by_type"].items():
        print(f"  {k:24s} rows={v['rows']:3d} impact={v['impact']:>12,.2f}")
