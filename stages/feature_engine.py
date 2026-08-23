"""
stages/feature_engine.py — Stage 1.5: Feature Engineering (deterministic catalog).

Rules fire based on EDA evidence; every applied rule is ledger-logged with the
evidence that triggered it. The Planner may later suggest extras (also logged).
"""
from __future__ import annotations

import pandas as pd

from core.state import RunState

LAGS = (1, 7, 30)
ROLL_WINDOWS = (7, 30)


def _time_features(df: pd.DataFrame, date_col: str, target_col: str,
                   freq: str) -> tuple[pd.DataFrame, list[str]]:
    applied: list[str] = []
    df = df.sort_values(date_col).reset_index(drop=True)
    d = df[date_col]
    df["dow"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    applied += ["dow", "month", "is_month_end"]

    if freq == "daily":
        for lag in LAGS:
            if len(df) > lag + 5:
                df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
                applied.append(f"{target_col}_lag{lag}")
        for w in ROLL_WINDOWS:
            if len(df) > w + 5:
                df[f"{target_col}_rollmean{w}"] = df[target_col].rolling(w).mean()
                df[f"{target_col}_rollstd{w}"] = df[target_col].rolling(w).std()
                applied += [f"{target_col}_rollmean{w}", f"{target_col}_rollstd{w}"]
    elif freq in ("weekly", "monthly"):
        lag = 1
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
        applied.append(f"{target_col}_lag{lag}")

    df[f"{target_col}_pct_change"] = df[target_col].pct_change()
    applied.append(f"{target_col}_pct_change")
    return df, applied


def _rfm_features(df: pd.DataFrame, date_col: str, id_col: str,
                  target_col: str) -> tuple[pd.DataFrame, list[str]]:
    """Customer-level RFM merged back onto rows."""
    ref_date = df[date_col].max()
    grp = df.groupby(id_col)
    rfm = pd.DataFrame({
        "rfm_recency_days": (ref_date - grp[date_col].max()).dt.days,
        "rfm_frequency": grp.size(),
        "rfm_monetary": grp[target_col].sum(),
    }).reset_index()
    df = df.merge(rfm, on=id_col, how="left")
    return df, ["rfm_recency_days", "rfm_frequency", "rfm_monetary"]


def _price_qty_features(df: pd.DataFrame, price_col: str, qty_col: str,
                        target_col: str) -> tuple[pd.DataFrame, list[str]]:
    applied: list[str] = []
    df["revenue_per_unit"] = df[target_col] / df[qty_col].replace(0, pd.NA)
    applied.append("revenue_per_unit")
    df["expected_revenue"] = df[price_col] * df[qty_col]
    df["billed_vs_expected_delta"] = df[target_col] - df["expected_revenue"]
    applied += ["expected_revenue", "billed_vs_expected_delta"]  # leakage signal
    return df, applied


def _find_col(cols: list[str], *keys: str, numeric_in: "pd.DataFrame | None" = None) -> str | None:
    """First column whose name contains a key. If numeric_in is given, the
    column must ALSO be numeric dtype — a name like 'tariff_code' holding
    strings ('RES-STD') is a label, not a price, and must not match."""
    for key in keys:
        for c in cols:
            if key in c.lower():
                if numeric_in is not None and not pd.api.types.is_numeric_dtype(numeric_in[c]):
                    continue
                return c
    return None


def run_feature_engineering(state: RunState) -> RunState:
    df = state.raw_df.copy()
    eda = state.eda_report
    date_cols = state.ingest_report["date_columns"]
    target = eda.get("target_guess")
    applied: dict[str, list[str]] = {}
    evidence: list[str] = []

    ts = eda.get("timeseries") or {}
    if date_cols and target and ts:
        df, feats = _time_features(df, date_cols[0], target, ts.get("frequency", ""))
        applied["time"] = feats
        evidence.append("eda.timeseries.frequency")

    cols = df.columns.tolist()
    id_col = _find_col(cols, "customer", "account", "client", "meter", "user_id")
    if id_col and date_cols and target:
        df, feats = _rfm_features(df, date_cols[0], id_col, target)
        applied["rfm"] = feats
        evidence.append("eda.categorical(id column present)")

    price_col = _find_col(cols, "price", "rate", "tariff", "unit_cost", numeric_in=df)
    qty_col = _find_col(cols, "qty", "quantity", "units", "volume", "kwh", "usage",
                        "consumption", numeric_in=df)
    if price_col and qty_col and target and price_col != target and qty_col != target:
        try:
            df, feats = _price_qty_features(df, price_col, qty_col, target)
            applied["price_qty"] = feats
            evidence.append("eda.numeric(price+quantity columns present)")
        except (TypeError, ValueError) as exc:  # never let one rule kill the run
            evidence.append(f"price_qty rule skipped: {exc}")

    if "price_qty" not in applied and qty_col and target and qty_col != target:
        try:
            df["revenue_per_unit"] = df[target] / df[qty_col].replace(0, pd.NA)
            applied["qty_only"] = ["revenue_per_unit"]
            evidence.append("eda.numeric(quantity column present; no numeric price)")

            # No numeric price, but a categorical tariff/plan label? Infer the
            # unit rate per label (median revenue_per_unit of normal rows) and
            # rebuild the expected-revenue leakage signal from it.
            label_col = _find_col(cols, "tariff", "plan", "rate", "price_code",
                                  "product_code")
            if (label_col and not pd.api.types.is_numeric_dtype(df[label_col])
                    and df[label_col].nunique() <= max(20, int(len(df) ** 0.5))):
                rpu = df["revenue_per_unit"]
                valid = rpu.notna() & (rpu > 0)
                grp_rate = (df.loc[valid].groupby(label_col)["revenue_per_unit"]
                            .median().rename("inferred_unit_rate"))
                df = df.merge(grp_rate, on=label_col, how="left")
                df["expected_revenue"] = df["inferred_unit_rate"] * df[qty_col]
                df["billed_vs_expected_delta"] = df[target] - df["expected_revenue"]
                applied["inferred_price"] = ["inferred_unit_rate", "expected_revenue",
                                             "billed_vs_expected_delta"]
                evidence.append(
                    f"eda.categorical('{label_col}' label + quantity → unit rate "
                    "inferred per label group; expected_revenue rebuilt)")
        except (TypeError, ValueError):
            pass

    state.feature_df = df
    state.feature_report = {
        "applied_rules": applied,
        "n_features_added": sum(len(v) for v in applied.values()),
        "detected_roles": {"id_col": id_col, "price_col": price_col, "qty_col": qty_col},
    }

    state.ledger.log(
        stage="feature_engineering",
        agent="deterministic",
        decision=f"Applied {len(applied)} rule group(s), "
                 f"{state.feature_report['n_features_added']} features added",
        reasoning="; ".join(f"{k}: {len(v)} feats" for k, v in applied.items())
                  or "no rules triggered (no date/target/id/price signals)",
        evidence=evidence or ["eda_report"],
        data=state.feature_report["detected_roles"],
    )
    return state
