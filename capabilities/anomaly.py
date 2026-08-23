"""
capabilities/anomaly.py — Anomaly detection: 5-detector registry + consensus voting.

Detectors vote row-wise; consensus tiers:
    high   = flagged by >= 3 detectors
    medium = flagged by 2
    review = flagged by 1
Per-anomaly feature attribution: z-scores of the flagged row's features vs
column distributions — "which features drove this flag".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from core.state import RunState

try:
    from statsmodels.tsa.seasonal import STL
    HAS_STL = True
except ImportError:
    HAS_STL = False

CONTAMINATION = 0.01        # expected anomaly fraction (POC default)
TIERS = {"high": 3, "medium": 2, "review": 1}


# ----------------------------------------------------------------------------
# detectors — each returns a boolean flag Series aligned to X.index
# ----------------------------------------------------------------------------

def _det_isolation_forest(X: pd.DataFrame) -> pd.Series:
    m = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=0)
    return pd.Series(m.fit_predict(X) == -1, index=X.index)


def _det_one_class_svm(X: pd.DataFrame) -> pd.Series:
    Xs = StandardScaler().fit_transform(X)
    m = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale")
    return pd.Series(m.fit_predict(Xs) == -1, index=X.index)


def _det_lof(X: pd.DataFrame) -> pd.Series:
    n_neighbors = min(20, max(5, len(X) // 50))
    m = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=CONTAMINATION)
    return pd.Series(m.fit_predict(X) == -1, index=X.index)


def _det_zscore_iqr(X: pd.DataFrame) -> pd.Series:
    """Univariate extremes: |z| > 4 OR outside 3×IQR fence in ANY feature."""
    flags = pd.Series(False, index=X.index)
    for col in X.columns:
        s = X[col]
        std = s.std()
        if std > 0:
            flags |= ((s - s.mean()).abs() / std) > 4
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            flags |= (s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)
    return flags


def _det_stl_residual(df: pd.DataFrame, date_col: str, target_col: str,
                      period: int) -> pd.Series:
    """Time-aware: rows on dates whose de-seasonalized daily residual is extreme."""
    ts = df.groupby(date_col)[target_col].sum().sort_index()
    if not HAS_STL or len(ts) < 2 * period + 1:
        return pd.Series(False, index=df.index)
    res = STL(ts.astype(float), period=period, robust=True).fit().resid
    std = res.std()
    if std == 0:
        return pd.Series(False, index=df.index)
    bad_dates = set(res.index[(res.abs() / std) > 3.5])
    return df[date_col].isin(bad_dates)


# ----------------------------------------------------------------------------
# attribution
# ----------------------------------------------------------------------------

def _attribute(X: pd.DataFrame, idx, top_n: int = 3) -> list[str]:
    """Features whose value on this row is most extreme vs the column."""
    row = X.loc[idx]
    scores = {}
    for col in X.columns:
        std = X[col].std()
        if std > 0:
            scores[col] = abs(float(row[col] - X[col].mean()) / std)
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [f"{c} (z={z:.1f})" for c, z in top if z > 2]


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------

def run_anomaly_capability(state: RunState, detectors: list[str] | None = None) -> dict:
    cm = state.column_map
    df = state.feature_df if state.feature_df is not None else state.raw_df
    date_col, target_col = cm.get("date_column"), cm["target_column"]
    period = ((state.eda_report.get("timeseries") or {}).get("seasonality") or {}
              ).get("period", 7)

    # feature matrix: numeric, exclude engineered rolling/lag noise for detection
    base_cols = [c for c in state.ingest_report["numeric_columns"] if c in df.columns]
    extra = [c for c in ("billed_vs_expected_delta", "revenue_per_unit") if c in df.columns]
    X = df[base_cols + extra].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    available = {
        "isolation_forest": lambda: _det_isolation_forest(X),
        "one_class_svm": lambda: _det_one_class_svm(X),
        "lof": lambda: _det_lof(X),
        "zscore_iqr": lambda: _det_zscore_iqr(X),
        "stl_residual": (lambda: _det_stl_residual(df, date_col, target_col, period))
                        if date_col else None,
    }
    chosen = detectors or [k for k, v in available.items() if v is not None]
    votes = pd.DataFrame(index=df.index)
    errors: dict[str, str] = {}
    for name in chosen:
        fn = available.get(name)
        if fn is None:
            errors[name] = "not applicable (no date column)" if name == "stl_residual" else "unknown detector"
            continue
        try:
            votes[name] = fn()
        except Exception as e:  # noqa: BLE001
            errors[name] = f"{type(e).__name__}: {e}"

    vote_count = votes.sum(axis=1)
    tier = pd.Series("none", index=df.index)
    tier[vote_count >= TIERS["review"]] = "review"
    tier[vote_count >= TIERS["medium"]] = "medium"
    tier[vote_count >= TIERS["high"]] = "high"

    flagged_idx = vote_count[vote_count >= 1].sort_values(ascending=False).index
    flagged = []
    for idx in flagged_idx[:200]:
        rec = {
            "row_index": int(idx),
            "tier": tier.loc[idx],
            "votes": int(vote_count.loc[idx]),
            "voted_by": [c for c in votes.columns if bool(votes.loc[idx, c])],
            "attribution": _attribute(X, idx),
        }
        if cm.get("id_column") and cm["id_column"] in df.columns:
            rec["entity"] = str(df.loc[idx, cm["id_column"]])
        if date_col:
            rec["date"] = str(df.loc[idx, date_col])
        rec[target_col] = float(df.loc[idx, target_col]) if pd.notna(df.loc[idx, target_col]) else None
        flagged.append(rec)

    out = {
        "detectors_run": list(votes.columns),
        "roster": {
            name: ({"status": "ran",
                    "votes_cast": int(votes[name].sum()) if name in votes.columns else 0}
                   if name in votes.columns else
                   {"status": "skipped",
                    "reason": errors.get(name, "not applicable to this data")})
            for name in available
        },
        "detector_errors": errors,
        "counts": {t: int((tier == t).sum()) for t in ("high", "medium", "review")},
        "flagged": flagged,
    }
    state.results["anomaly"] = out
    state.ledger.log(
        stage="anomaly", agent="deterministic",
        decision=(f"{out['counts']['high']} high / {out['counts']['medium']} medium / "
                  f"{out['counts']['review']} review-tier anomalies "
                  f"({len(votes.columns)} detectors voting)"),
        reasoning=(f"consensus voting across {list(votes.columns)}; "
                   f"high tier requires >= {TIERS['high']} detector agreement"
                   + (f"; detector errors: {errors}" if errors else "")),
        evidence=["anomaly.votes", "eda.numeric"],
        data={"counts": out["counts"]},
    )
    return out
