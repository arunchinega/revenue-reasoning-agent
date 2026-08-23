"""
capabilities/segmentation.py — Customer segmentation (RFM + KMeans, silhouette-picked k)
capabilities: what-if scenario simulation, recommendation synthesis.
Grouped in one module: each is compact and they share downstream wiring.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from core.llm import call_json, call_text
from core.state import RunState


# ============================================================================
# SEGMENTATION
# ============================================================================

def run_segmentation_capability(state: RunState, use_llm: bool = True) -> dict:
    cm = state.column_map
    df = state.feature_df if state.feature_df is not None else state.raw_df
    id_col = cm.get("id_column")
    if not id_col or id_col not in df.columns:
        out = {"error": "segmentation requires an id column"}
        state.results["segment"] = out
        return out

    rfm_cols = [c for c in ("rfm_recency_days", "rfm_frequency", "rfm_monetary")
                if c in df.columns]
    if len(rfm_cols) < 2:
        out = {"error": "RFM features unavailable"}
        state.results["segment"] = out
        return out

    ent = df.groupby(id_col)[rfm_cols].first().dropna()
    Xs = StandardScaler().fit_transform(ent)

    # k chosen by silhouette sweep — the Critic-style decision, logged
    best_k, best_score = 2, -1.0
    for k in range(2, min(8, len(ent) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score = k, score
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(Xs)
    ent = ent.assign(segment_id=labels)

    profiles = []
    for sid, grp in ent.groupby("segment_id"):
        prof = {"segment_id": int(sid), "size": int(len(grp)),
                **{c: round(float(grp[c].mean()), 1) for c in rfm_cols}}
        profiles.append(prof)
    profiles.sort(key=lambda p: p.get("rfm_monetary", 0), reverse=True)

    # persona naming: Gemma if available, deterministic tags otherwise
    med_rec = ent["rfm_recency_days"].median() if "rfm_recency_days" in ent else 0
    med_mon = ent["rfm_monetary"].median() if "rfm_monetary" in ent else 0
    for p in profiles:
        hi_val = p.get("rfm_monetary", 0) >= med_mon
        recent = p.get("rfm_recency_days", 0) <= med_rec
        p["persona"] = (
            "High-value, active" if hi_val and recent else
            "High-value, lapsing — churn risk" if hi_val else
            "Low-value, active" if recent else "Low-value, dormant"
        )
    if use_llm:
        r = call_text(
            role="narrator",
            system="Name customer segments. One short vivid persona name per "
                   "segment, given its RFM profile. Return one line per segment: "
                   "'<segment_id>: <persona name>'. Nothing else.",
            user=json.dumps(profiles),
        )
        if r.ok and not r.used_fallback:
            for line in r.content.splitlines():
                if ":" in line:
                    sid, name = line.split(":", 1)
                    for p in profiles:
                        if str(p["segment_id"]) == sid.strip().lstrip("- "):
                            p["persona"] = name.strip()

    seg_map = ent["segment_id"].to_dict()
    out = {"k": best_k, "silhouette": round(best_score, 3), "profiles": profiles,
           "entity_segments": {str(k): int(v) for k, v in list(seg_map.items())[:500]}}
    state.results["segment"] = out
    state.ledger.log(
        stage="segmentation", agent="deterministic",
        decision=f"k={best_k} segments over {len(ent)} entities",
        reasoning=f"silhouette sweep 2-8 picked k={best_k} (score {best_score:.3f}); "
                  f"profiles: " + "; ".join(
                      f"seg{p['segment_id']} n={p['size']} '{p['persona']}'"
                      for p in profiles),
        evidence=["feature.rfm", "segment.silhouette"],
        confidence=round(float(best_score), 3),
    )
    return out


# ============================================================================
# WHAT-IF
# ============================================================================

def run_whatif_capability(state: RunState, scenarios: list[dict] | None = None) -> dict:
    """Perturb inputs, re-run through the accepted forecast. POC assumption
    (declared in ledger): linear pass-through elasticity unless overridden."""
    fc = state.results.get("forecasting") or {}
    base = fc.get("forecast")
    if not base:
        out = {"error": "what-if requires an accepted forecast first"}
        state.results["whatif"] = out
        return out
    base = np.array(base, dtype=float)

    scenarios = scenarios or [
        {"name": "price +5%", "revenue_multiplier": 1.05},
        {"name": "price -5%", "revenue_multiplier": 0.95},
        {"name": "volume -10%", "revenue_multiplier": 0.90},
        {"name": "recover leakage",
         "revenue_add_total": (state.results.get("leakage") or {}
                               ).get("total_impact_estimate", 0)},
    ]
    rows = []
    for sc in scenarios:
        adj = base * float(sc.get("revenue_multiplier", 1.0))
        add = float(sc.get("revenue_add_total") or 0)
        if add:
            adj = adj + add / len(adj)
        rows.append({
            "scenario": sc["name"],
            "total": round(float(adj.sum()), 2),
            "delta_vs_baseline": round(float(adj.sum() - base.sum()), 2),
            "delta_pct": round(float((adj.sum() / base.sum() - 1) * 100), 2),
        })

    out = {"baseline_total": round(float(base.sum()), 2),
           "horizon_days": len(base), "scenarios": rows,
           "assumption": "linear pass-through elasticity (POC default)"}
    state.results["whatif"] = out
    state.ledger.log(
        stage="whatif", agent="deterministic",
        decision=f"{len(rows)} scenario(s) vs baseline {out['baseline_total']:,.0f}",
        reasoning="ASSUMPTION (explicit): linear pass-through elasticity; "
                  + "; ".join(f"{r['scenario']}: {r['delta_pct']:+.1f}%" for r in rows),
        evidence=["forecasting.forecast", "leakage.total_impact_estimate"],
    )
    return out


# ============================================================================
# RECOMMENDATIONS
# ============================================================================

def run_recommend_capability(state: RunState, use_llm: bool = True) -> dict:
    """Synthesize ranked actions — every one must trace to a cited finding."""
    findings = {
        "leakage": {k: (state.results.get("leakage") or {}).get(k)
                    for k in ("total_impact_estimate", "candidate_count", "rules_fired")},
        "forecast": {k: (state.results.get("forecasting") or {}).get(k)
                     for k in ("verdict", "winner", "metrics")},
        "anomaly_counts": (state.results.get("anomaly") or {}).get("counts"),
        "rca_hypotheses": (state.results.get("rca") or {}).get("hypotheses"),
        "segments": (state.results.get("segment") or {}).get("profiles"),
        "whatif": (state.results.get("whatif") or {}).get("scenarios"),
    }

    def _fallback() -> dict:
        recs = []
        lk = findings["leakage"]
        if lk and lk.get("total_impact_estimate"):
            recs.append({
                "action": f"Investigate top leakage candidates "
                          f"(estimated recoverable {lk['total_impact_estimate']:,.0f})",
                "expected_impact": lk["total_impact_estimate"],
                "effort": "low", "confidence": 0.8,
                "traces_to": "leakage.total_impact_estimate",
            })
        seg = findings["segments"] or []
        churny = [p for p in seg if "churn" in str(p.get("persona", "")).lower()
                  or "lapsing" in str(p.get("persona", "")).lower()]
        for p in churny[:1]:
            recs.append({
                "action": f"Retention outreach to segment {p['segment_id']} "
                          f"('{p['persona']}', {p['size']} customers)",
                "expected_impact": p.get("rfm_monetary", 0) * p.get("size", 0) * 0.1,
                "effort": "medium", "confidence": 0.6,
                "traces_to": f"segment.profiles[{p['segment_id']}]",
            })
        for h in (findings["rca_hypotheses"] or [])[:1]:
            recs.append({
                "action": f"Validate root-cause hypothesis: {h['hypothesis']}",
                "expected_impact": None, "effort": "low",
                "confidence": h.get("confidence", 0.5), "traces_to": "rca.hypotheses",
            })
        return {"recommendations": recs,
                "reasoning": "deterministic synthesis from findings"}

    if use_llm:
        llm = call_json(
            role="reasoner",
            system=(
                "You are a revenue advisor. Produce 3-5 ranked recommended actions "
                "STRICTLY grounded in the findings JSON — every action must cite a "
                "finding in traces_to, and expected_impact must come from the "
                "findings, never invented. Respond ONLY with JSON: "
                '{"recommendations": [{"action": str, "expected_impact": float|null, '
                '"effort": "low"|"medium"|"high", "confidence": float, '
                '"traces_to": str}], "reasoning": str}'
            ),
            user=json.dumps(findings, default=str),
            required_keys=("recommendations",),
            fallback=_fallback,
        )
        parsed = llm.parsed
        agent = "deterministic" if llm.used_fallback else "llama-3.1-8b"
    else:
        parsed = _fallback()
        agent = "deterministic"

    out = {"recommendations": parsed.get("recommendations", [])}
    state.results["recommend"] = out
    state.ledger.log(
        stage="recommend", agent=agent,
        decision=f"{len(out['recommendations'])} recommended action(s)",
        reasoning=parsed.get("reasoning", ""),
        evidence=[r.get("traces_to", "") for r in out["recommendations"][:5]],
    )
    return out
