"""
stages/eda_profiler.py — Stage 1: EDA / Auto-Profiling (deterministic, no LLM).

Emits eda_report.json — the evidence store every downstream agent cites.
Evidence keys are dotted paths (e.g. "eda.seasonality.weekly") so ledger
entries can reference exact facts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.state import RunState

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:  # graceful degradation — ts diagnostics skipped
    HAS_STATSMODELS = False


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _profile_structure(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns_detail": {
            col: {
                "dtype": str(df[col].dtype),
                "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
                "cardinality": int(df[col].nunique(dropna=True)),
            }
            for col in df.columns
        },
    }


def _profile_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> dict:
    out: dict = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr) if iqr > 0 else pd.Series(False, index=s.index)
        out[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "skew": round(float(s.skew()), 3),
            "kurtosis": round(float(s.kurtosis()), 3),
            "outlier_pct_iqr": round(float(outlier_mask.mean()) * 100, 2),
            "zero_pct": round(float((s == 0).mean()) * 100, 2),
            "negative_pct": round(float((s < 0).mean()) * 100, 2),
        }
    return out


def _profile_categorical(df: pd.DataFrame, numeric_cols: list[str],
                         date_cols: list[str], max_topk: int = 8) -> dict:
    out: dict = {}
    for col in df.columns:
        if col in numeric_cols or col in date_cols:
            continue
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        vc = s.value_counts(normalize=True)
        # normalized entropy: 1.0 = uniform, 0.0 = single value
        probs = vc.values
        entropy = float(-(probs * np.log2(probs)).sum()) if len(probs) > 1 else 0.0
        max_entropy = float(np.log2(len(probs))) if len(probs) > 1 else 1.0
        out[col] = {
            "cardinality": int(s.nunique()),
            "top_values": vc.head(max_topk).round(4).to_dict(),
            "entropy_normalized": round(entropy / max_entropy, 3) if max_entropy else 0.0,
            "segmentation_viable": 2 <= s.nunique() <= 50,
        }
    return out


def _infer_frequency(dates: pd.Series) -> tuple[str, float]:
    """Return (label, median_gap_days). Label: daily/weekly/monthly/irregular."""
    d = dates.dropna().sort_values().drop_duplicates()
    if len(d) < 3:
        return "unknown", float("nan")
    gaps = d.diff().dropna().dt.total_seconds() / 86400.0
    med = float(gaps.median())
    if med <= 1.5:
        label = "daily"
    elif med <= 8:
        label = "weekly"
    elif med <= 32:
        label = "monthly"
    else:
        label = "irregular"
    # irregularity check: high gap variance → irregular regardless
    if len(gaps) > 5 and gaps.std() > 2 * max(gaps.median(), 1e-9):
        label = "irregular"
    return label, round(med, 2)


_SEASONAL_PERIOD = {"daily": 7, "weekly": 52, "monthly": 12}


def _timeseries_diagnostics(df: pd.DataFrame, date_col: str, target_col: str) -> dict:
    """Trend/seasonality/stationarity on the (date, target) pair."""
    ts = (
        df[[date_col, target_col]]
        .dropna()
        .sort_values(date_col)
        .groupby(date_col, as_index=True)[target_col]
        .sum()
    )
    freq_label, median_gap = _infer_frequency(ts.index.to_series())
    diag: dict = {
        "date_column": date_col,
        "target_column": target_col,
        "points": int(len(ts)),
        "date_min": str(ts.index.min()),
        "date_max": str(ts.index.max()),
        "frequency": freq_label,
        "median_gap_days": median_gap,
        "date_gaps": None,
        "trend_direction": None,
        "seasonality": {},
        "stationary_adf": None,
        "adf_pvalue": None,
    }
    # gap count vs expected regular grid
    if freq_label in ("daily", "weekly", "monthly") and len(ts) > 2:
        expected = (ts.index.max() - ts.index.min()).days / max(median_gap, 1e-9) + 1
        diag["date_gaps"] = max(int(round(expected - len(ts))), 0)

    # simple trend: linear fit slope sign, normalized
    if len(ts) >= 10:
        x = np.arange(len(ts), dtype=float)
        slope = float(np.polyfit(x, ts.values.astype(float), 1)[0])
        rng = float(ts.max() - ts.min()) or 1.0
        norm_slope = slope * len(ts) / rng
        diag["trend_direction"] = (
            "increasing" if norm_slope > 0.1 else
            "decreasing" if norm_slope < -0.1 else "flat"
        )
        diag["trend_strength_norm"] = round(norm_slope, 3)

    if HAS_STATSMODELS and len(ts) >= 30:
        period = _SEASONAL_PERIOD.get(freq_label)
        if period and len(ts) >= 2 * period + 1:
            try:
                res = STL(ts.astype(float), period=period, robust=True).fit()
                var_resid = float(np.var(res.resid))
                var_seas = float(np.var(res.seasonal + res.resid))
                strength = max(0.0, 1 - var_resid / var_seas) if var_seas > 0 else 0.0
                diag["seasonality"] = {
                    "period": period,
                    "strength": round(strength, 3),
                    "detected": strength >= 0.3,
                }
            except Exception:  # noqa: BLE001 — diagnostics are best-effort
                pass
        try:
            adf_stat, pvalue, *_ = adfuller(ts.dropna().astype(float), autolag="AIC")
            diag["stationary_adf"] = bool(pvalue < 0.05)
            diag["adf_pvalue"] = round(float(pvalue), 4)
        except Exception:  # noqa: BLE001
            pass
    return diag


def _guess_target(df: pd.DataFrame, numeric_cols: list[str]) -> str | None:
    """Heuristic pre-guess of the revenue/target column (Stage 2.5 refines it)."""
    priority = ("revenue", "amount", "sales", "billed", "total", "value", "price", "charge")
    lowered = {c.lower(): c for c in numeric_cols}
    for key in priority:
        for low, orig in lowered.items():
            if key in low:
                return orig
    # fallback: numeric col with highest variance (excluding likely IDs)
    best, best_var = None, -1.0
    for col in numeric_cols:
        if df[col].nunique() == len(df):  # likely an ID
            continue
        v = float(df[col].var() or 0)
        if v > best_var:
            best, best_var = col, v
    return best


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def run_eda(state: RunState) -> RunState:
    df = state.raw_df
    numeric_cols = state.ingest_report["numeric_columns"]
    date_cols = state.ingest_report["date_columns"]

    report: dict = {
        "structure": _profile_structure(df),
        "numeric": _profile_numeric(df, numeric_cols),
        "categorical": _profile_categorical(df, numeric_cols, date_cols),
        "target_guess": _guess_target(df, numeric_cols),
        "timeseries": None,
    }

    if date_cols and report["target_guess"]:
        report["timeseries"] = _timeseries_diagnostics(
            df, date_cols[0], report["target_guess"]
        )

    state.eda_report = report
    state.save_report("eda_report", report)

    # ledger summary with cited evidence
    ev = ["eda.structure.rows"]
    bits = [f"{report['structure']['rows']} rows profiled"]
    ts = report["timeseries"]
    if ts:
        bits.append(f"time series: {ts['frequency']}, {ts['points']} points")
        ev.append("eda.timeseries.frequency")
        seas = ts.get("seasonality") or {}
        if seas.get("detected"):
            bits.append(f"seasonality period={seas['period']} strength={seas['strength']}")
            ev.append("eda.timeseries.seasonality")
        if ts.get("stationary_adf") is not None:
            bits.append(f"ADF stationary={ts['stationary_adf']} (p={ts['adf_pvalue']})")
            ev.append("eda.timeseries.stationary_adf")
    high_missing = [
        c for c, d in report["structure"]["columns_detail"].items()
        if d["missing_pct"] > 20
    ]
    if high_missing:
        bits.append(f"high-missing cols: {high_missing}")
        ev.append("eda.structure.columns_detail")

    state.ledger.log(
        stage="eda",
        agent="deterministic",
        decision="EDA profile complete → eda_report.json",
        reasoning="; ".join(bits),
        evidence=ev,
        data={"target_guess": report["target_guess"]},
    )
    return state
