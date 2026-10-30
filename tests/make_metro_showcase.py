"""
tests/make_metro_showcase.py — demo-grade synthetic METRO FARE dataset
(scenario 2 of the infra series). Same contract as make_utilities_showcase:
every capability has something real to find, and a manifest records it.

Domain signals: fare, trip, line, station, concession  →  'transit' profile.
Column roles:  account_id → id · fare_revenue → target · fare_rate → price ·
               trip_units → quantity · concession_discount_pct → discount.

Writes <path>.csv and <path>.manifest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

START = "2025-01-01"
FARES = {                      # INR per trip, constant per fare class
    "regular": 32.0,
    "corporate_pass": 28.0,
    "student": 16.0,
    "senior": 20.0,
    "tourist": 60.0,
}
BASE_DISCOUNT = {"regular": 0.0, "corporate_pass": 10.0, "student": 0.0,
                 "senior": 0.0, "tourist": 0.0}   # concession applied on top
LINES = ["L1_Red", "L2_Blue", "L3_Green", "L4_Purple", "L5_Orange"]
ZONES = ["North", "South", "East", "West", "Central"]
RCA_ZONE = "East"
EXT_LINE = "L4_Purple"


def make_metro_showcase(path: str = "demo_metro_showcase.csv", days: int = 548,
                        n_accounts: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START, periods=days, freq="D")
    change_day = pd.Timestamp("2025-11-15")              # L4 extension opens
    rca_start = pd.Timestamp("2026-02-10")               # East zone partial shutdown
    rca_end = rca_start + pd.Timedelta(days=29)

    # archetypes: name, fare class, trips/day band, weight
    archetypes = [
        ("daily_commuter", "regular", (1.6, 2.4), 0.30),
        ("occasional_rider", "regular", (0.2, 0.7), 0.18),
        ("corporate_fleet", "corporate_pass", (40, 160), 0.14),   # one account = many staff
        ("school_group", "student", (15, 60), 0.14),
        ("senior_card", "senior", (0.4, 1.2), 0.12),
        ("tour_operator", "tourist", (8, 40), 0.12),
    ]
    w = np.array([a[3] for a in archetypes]); w /= w.sum()

    accounts = []
    for i in range(n_accounts):
        a = archetypes[rng.choice(len(archetypes), p=w)]
        zone = rng.choice(ZONES)
        line = rng.choice(LINES)
        r = rng.random()
        if r < 0.72:
            s, e = 0, days
        elif r < 0.86:
            s, e = 0, int(rng.integers(days // 2, days - 30))
        else:
            s, e = int(rng.integers(30, days // 2)), days
        accounts.append(dict(
            account_id=f"ACC{i:05d}", arch=a[0], fare_class=a[1], zone=zone,
            line=line, base=float(rng.uniform(*a[2])), start=s, end=e,
            noise=0.10 if a[0] in ("corporate_fleet", "school_group") else 0.25))

    doy = np.array([d.dayofyear for d in dates])
    dow = np.array([d.dayofweek for d in dates])
    month = np.array([d.month for d in dates])
    elapsed = np.arange(days)
    monsoon = np.where(np.isin(month, [7, 8, 9]), 0.88, 1.0)          # monsoon dip
    festival = 1 + 0.15 * np.exp(-((doy - 290) / 12) ** 2)            # Oct festival peak
    trend = 1 + 0.00030 * elapsed                                     # ~11%/yr ridership growth

    frames = []
    for a in accounts:
        idx = np.arange(a["start"], a["end"])
        if a["fare_class"] in ("regular", "corporate_pass", "student", "senior"):
            weekly = np.where(dow[idx] >= 5, 0.55, 1.0)                # weekend drop
            if a["fare_class"] == "student":
                weekly = np.where(dow[idx] >= 5, 0.25, 1.0)            # schools shut
        else:
            weekly = np.where(dow[idx] >= 5, 1.45, 1.0)                # tourists weekend up
        ext = np.where((dates[idx] > change_day) & (a["line"] == EXT_LINE), 1.30, 1.0)
        trips = a["base"] * weekly * monsoon[idx] * festival[idx] * trend[idx] * ext \
            * rng.normal(1, a["noise"], len(idx))
        # small riders: integer trips via Poisson so 0-trip days happen naturally
        if a["base"] < 5:
            trips = rng.poisson(np.clip(trips, 0.01, None)).astype(float)
        else:
            trips = np.round(np.clip(trips, 0, None))
        frames.append(pd.DataFrame({
            "travel_date": dates[idx].date,
            "account_id": a["account_id"],
            "fare_class": a["fare_class"], "metro_line": a["line"], "zone": a["zone"],
            "fare_rate": FARES[a["fare_class"]],
            "trip_units": trips,
            "concession_discount_pct": BASE_DISCOUNT[a["fare_class"]],
            "_arch": a["arch"],
        }))
    df = pd.concat(frames, ignore_index=True)
    df["travel_date"] = pd.to_datetime(df["travel_date"])

    # RCA: partial shutdown of East zone stations
    rca_mask = (df["zone"] == RCA_ZONE) & df["travel_date"].between(rca_start, rca_end)
    df.loc[rca_mask, "trip_units"] = np.round(df.loc[rca_mask, "trip_units"] * 0.45)

    def recompute(ix):
        df.loc[ix, "fare_revenue"] = (df.loc[ix, "trip_units"] * df.loc[ix, "fare_rate"]
                                      * (1 - df.loc[ix, "concession_discount_pct"] / 100)).round(2)
    df["fare_revenue"] = 0.0
    recompute(df.index)
    df = df.sort_values(["travel_date", "account_id"]).reset_index(drop=True)
    rca_mask = (df["zone"] == RCA_ZONE) & df["travel_date"].between(rca_start, rca_end)

    manifest = {"defects": [], "summary": {}}

    def plant(kind, i, impact=0.0, note=""):
        manifest["defects"].append({
            "type": kind, "account_id": df.at[i, "account_id"],
            "travel_date": str(df.at[i, "travel_date"].date()),
            "impact": round(float(impact), 2), "note": note})

    pool = df.index[(~rca_mask) & (df["trip_units"] >= 10)].to_numpy().copy()   # meaningful rows
    rng.shuffle(pool)
    cur = 0

    def take(n):
        nonlocal cur
        out = pool[cur:cur + n]; cur += n
        return out

    # leakage: under-collected (gates let trips through, revenue short)
    for i in take(24):
        correct = df.at[i, "fare_revenue"]
        f = rng.uniform(0.45, 0.70)
        df.at[i, "fare_revenue"] = round(correct * f, 2)
        plant("under_collected", i, correct - df.at[i, "fare_revenue"], f"collected {f:.0%} of due")

    # leakage: trips recorded, zero revenue (gate fault / unsettled clearing)
    for i in take(15):
        correct = df.at[i, "fare_revenue"]
        df.at[i, "fare_revenue"] = 0.0
        plant("zero_revenue_with_trips", i, correct, "trips recorded, nothing settled")

    # leakage: wrong fare class applied (lower rate than the class's modal fare)
    for i in take(18):
        fc = df.at[i, "fare_class"]
        wrong = FARES[fc] * rng.choice([0.5, 0.6, 0.7, 0.8])
        correct = df.at[i, "fare_revenue"]
        df.at[i, "fare_rate"] = round(wrong, 2)
        recompute([i])
        plant("wrong_fare_applied", i, correct - df.at[i, "fare_revenue"],
              f"{fc} charged {wrong:.0f} instead of {FARES[fc]:.0f}")

    # leakage: concession abuse (discount far above policy)
    for i in take(10):
        correct = df.at[i, "fare_revenue"]
        d = float(rng.choice([65, 70, 80, 90]))
        df.at[i, "concession_discount_pct"] = d
        recompute([i])
        plant("concession_abuse", i, correct - df.at[i, "fare_revenue"], f"{d:.0f}% concession applied")

    # anomaly: ridership spikes on the biggest accounts (corporate / school)
    big = df.index[df["_arch"].isin(["corporate_fleet", "school_group"]) & (~rca_mask)
                   & (df["trip_units"] >= 20)].to_numpy().copy()
    big = np.setdiff1d(big, pool[:cur]); rng.shuffle(big); bc = 0

    def take_big(n):
        nonlocal bc
        out = big[bc:bc + n]; bc += n
        return out

    for i in take_big(14):
        m = rng.uniform(9.0, 14.0)
        df.at[i, "trip_units"] = round(df.at[i, "trip_units"] * m)
        recompute([i])
        plant("ridership_spike", i, 0.0, f"x{m:.1f} trips (gate counter fault / event)")

    # anomaly: stuck gate counter — 3 zero days then a catch-up upload
    for i in take_big(8):
        acc, d0 = df.at[i, "account_id"], df.at[i, "travel_date"]
        run = df.index[(df["account_id"] == acc) & df["travel_date"].between(d0, d0 + pd.Timedelta(days=3))]
        if len(run) < 4:
            continue
        run = [int(r) for r in run]
        total = float(df.loc[run, "trip_units"].sum())
        for r in run[:3]:
            df.at[r, "trip_units"] = 0.0
            recompute([r])
            plant("stuck_counter_zero", r, 0.0, "counter offline (context row; zero is not a population outlier)")
        j = run[3]
        df.at[j, "trip_units"] = round(total)
        recompute([j])
        plant("stuck_counter_catchup", j, 0.0, "batch upload of 4 days' trips")

    # leakage: duplicate settlement rows
    dup_src = np.setdiff1d(take(40), big)[:12]
    dups = df.loc[dup_src].copy()
    for i in dup_src:
        plant("duplicate_settlement", i, df.at[i, "fare_revenue"], "same account/date/amount settled twice")
    df = pd.concat([df, dups], ignore_index=True)

    miss = rng.choice(df.index, int(len(df) * 0.006), replace=False)
    df.loc[miss, "trip_units"] = np.nan

    df = df.sort_values(["travel_date", "account_id"], kind="stable").reset_index(drop=True)
    for d in manifest["defects"]:
        hits = df.index[(df["account_id"] == d["account_id"])
                        & (df["travel_date"] == pd.Timestamp(d["travel_date"]))]
        d["row_index"] = [int(h) for h in hits]

    arch = df.pop("_arch")
    df["travel_date"] = df["travel_date"].dt.date
    df = df[["travel_date", "account_id", "fare_class", "metro_line", "zone",
             "fare_rate", "trip_units", "concession_discount_pct", "fare_revenue"]]

    by_type = {}
    for d in manifest["defects"]:
        s = by_type.setdefault(d["type"], {"rows": 0, "impact": 0.0})
        s["rows"] += 1; s["impact"] = round(s["impact"] + d["impact"], 2)
    leak_types = ("under_collected", "zero_revenue_with_trips", "wrong_fare_applied",
                  "concession_abuse", "duplicate_settlement")
    manifest["summary"] = {
        "rows": int(len(df)), "accounts": n_accounts, "days": days,
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
        "planted_leakage_total": round(sum(by_type[t]["impact"] for t in leak_types if t in by_type), 2),
        "by_type": by_type,
        "forecasting": {
            "weekly_seasonality": "commuters/students weekend -45%/-75%; tourists weekend +45%",
            "yearly_seasonality": "monsoon dip Jul-Sep -12%; October festival peak +15%",
            "trend": "~11% per year",
            "changepoint": {"date": str(change_day.date()),
                            "effect": f"{EXT_LINE} extension opens: +30% trips on that line"},
        },
        "rca": {"zone": RCA_ZONE, "window": [str(rca_start.date()), str(rca_end.date())],
                "effect": "partial station shutdown: trips -55% in zone"},
        "segmentation": {"archetypes": arch.value_counts().to_dict(),
                         "lifecycle": "72% full-period, 14% churn, 14% late joiners"},
        "missing_trip_rows": int(len(miss)),
    }
    Path(path).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "demo_metro_showcase.csv"
    d = make_metro_showcase(out)
    man = json.loads(Path(out).with_suffix(".manifest.json").read_text())
    print(f"wrote {out}: {d.shape}")
    print("planted leakage total:", man["summary"]["planted_leakage_total"])
    for k, v in man["summary"]["by_type"].items():
        print(f"  {k:24s} rows={v['rows']:3d} impact={v['impact']:>12,.2f}")
