"""
stages/feature_analysis.py — Stage 1.6: Feature Analysis (deterministic).

Outputs feature_report additions: importance ranking, redundancy prune list,
leakage guard results — the "8 matter, 12 are noise" evidence for the Planner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

from core.state import RunState

CORR_REDUNDANT = 0.95      # |pairwise corr| above this → drop the later one
LEAKAGE_CORR = 0.999       # near-perfect corr with target → leakage suspect
TOP_K = 15


def run_feature_analysis(state: RunState) -> RunState:
    df = state.feature_df if state.feature_df is not None else state.raw_df
    target = state.eda_report.get("target_guess")

    numeric = df.select_dtypes(include="number")
    if target not in numeric.columns:
        state.ledger.log(
            stage="feature_analysis", agent="deterministic",
            decision="Skipped — no numeric target available",
            reasoning=f"target_guess={target!r} not numeric",
        )
        return state

    feats = [c for c in numeric.columns if c != target]
    X_full = numeric[feats]
    y_full = numeric[target]
    mask = y_full.notna()
    # sanitize: divisions by zero upstream (pct_change, per-unit ratios) yield inf
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    X = X_full[mask].fillna(X_full.median(numeric_only=True)).fillna(0)
    y = y_full[mask]

    # --- leakage guard: features (near-)identical to target ------------------
    # computed pairwise on RAW values (pre-imputation) so fill noise can't
    # dilute a near-perfect correlation below the threshold
    leakage_suspects: list[str] = []
    for c in list(feats):
        raw = X_full[c]
        pair = y_full.notna() & raw.notna()
        if pair.sum() < 3 or raw[pair].nunique() <= 1:
            continue
        corr = float(np.corrcoef(raw[pair], y_full[pair])[0, 1])
        if abs(corr) >= LEAKAGE_CORR:
            leakage_suspects.append(c)
    kept = [c for c in feats if c not in leakage_suspects]

    # --- redundancy prune (correlation-based, VIF-lite) ----------------------
    dropped_redundant: list[tuple[str, str, float]] = []
    if len(kept) > 1:
        corr_m = X[kept].corr().abs()
        upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
        for col in upper.columns:
            partner = upper[col].idxmax() if upper[col].notna().any() else None
            if partner and upper.loc[partner, col] >= CORR_REDUNDANT:
                dropped_redundant.append((col, partner, round(float(upper.loc[partner, col]), 4)))
        drop_set = {c for c, _, _ in dropped_redundant}
        kept = [c for c in kept if c not in drop_set]

    # --- importance: mutual info + quick RandomForest -------------------------
    importance: dict[str, dict[str, float]] = {}
    if kept and len(X) >= 20:
        Xk = X[kept]
        try:
            mi = mutual_info_regression(Xk, y, random_state=0)
            rf = RandomForestRegressor(n_estimators=60, random_state=0, n_jobs=-1,
                                       max_depth=8).fit(Xk, y)
            for i, c in enumerate(kept):
                importance[c] = {
                    "mutual_info": round(float(mi[i]), 4),
                    "rf_importance": round(float(rf.feature_importances_[i]), 4),
                }
        except Exception:  # noqa: BLE001 — analysis is best-effort
            pass

    ranked = sorted(importance.items(),
                    key=lambda kv: kv[1]["rf_importance"], reverse=True)
    top = [c for c, _ in ranked[:TOP_K]]
    noise = [c for c, v in ranked if v["rf_importance"] < 0.01]

    state.feature_report.update({
        "target": target,
        "leakage_suspects_dropped": leakage_suspects,
        "redundant_dropped": [
            {"dropped": a, "kept": b, "corr": r} for a, b, r in dropped_redundant
        ],
        "importance": importance,
        "top_features": top,
        "noise_features": noise,
    })
    state.save_report("feature_report", state.feature_report)

    state.ledger.log(
        stage="feature_analysis",
        agent="deterministic",
        decision=f"{len(top)} feature(s) matter, {len(noise)} are noise",
        reasoning=(
            f"leakage guard dropped {leakage_suspects or 'none'}; "
            f"redundancy pruned {len(dropped_redundant)}; "
            f"top by RF importance: {top[:5]}"
        ),
        evidence=["feature.importance", "feature.leakage_suspects_dropped"],
        data={"top_features": top[:10]},
    )
    return state
