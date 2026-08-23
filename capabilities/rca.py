"""
capabilities/rca.py — Root Cause Analysis.

Deterministic prep computes the evidence (contribution decomposition, period
deltas, anomaly overlaps); the LLM only synthesizes ranked hypotheses FROM that
evidence — it never invents numbers. Deterministic fallback produces hypotheses
straight from the decomposition ranking.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core.llm import call_json
from core.state import RunState


def _worst_period(ts: pd.Series, window: int = 30) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Find the window with the largest drop vs the preceding window."""
    roll = ts.rolling(window).sum()
    delta = roll - roll.shift(window)
    if delta.dropna().empty:
        mid = len(ts) // 2
        return ts.index[mid], ts.index[-1], 0.0
    end = delta.idxmin()
    start = end - pd.Timedelta(days=window - 1)
    return start, end, float(delta.min())


def _contribution_decomposition(df: pd.DataFrame, cm: dict, dim: str,
                                start, end, window: int) -> list[dict]:
    """Per-dimension-value change: focus window vs preceding window."""
    date_col, target = cm["date_column"], cm["target_column"]
    prev_start = start - pd.Timedelta(days=window)
    cur = df[(df[date_col] >= start) & (df[date_col] <= end)]
    prev = df[(df[date_col] >= prev_start) & (df[date_col] < start)]
    cur_g = cur.groupby(dim)[target].sum()
    prev_g = prev.groupby(dim)[target].sum()
    keys = sorted(set(cur_g.index) | set(prev_g.index))
    rows = []
    for k in keys:
        c, p = float(cur_g.get(k, 0)), float(prev_g.get(k, 0))
        rows.append({"value": str(k), "current": round(c, 2), "previous": round(p, 2),
                     "delta": round(c - p, 2)})
    rows.sort(key=lambda r: r["delta"])
    return rows


def run_rca_capability(state: RunState, use_llm: bool = True,
                       window: int = 30) -> dict:
    cm = state.column_map
    df = state.raw_df
    date_col, target = cm.get("date_column"), cm["target_column"]
    if not date_col:
        out = {"error": "RCA requires a date column"}
        state.results["rca"] = out
        return out

    ts = df.groupby(date_col)[target].sum().sort_index()
    start, end, drop = _worst_period(ts, window)

    # decompose across every low-cardinality dimension
    dims = [c for c, v in (state.eda_report.get("categorical") or {}).items()
            if v.get("segmentation_viable")]
    if cm.get("id_column") and cm["id_column"] in df.columns:
        dims = [d for d in dims if d != cm["id_column"]][:3]
        dims.append(cm["id_column"])
    decomp = {d: _contribution_decomposition(df, cm, d, start, end, window)
              for d in dims}

    # anomaly overlap inside the focus window
    overlap = []
    for rec in (state.results.get("anomaly") or {}).get("flagged", []):
        if rec.get("date") and start <= pd.Timestamp(rec["date"][:10]) <= end \
                and rec["tier"] in ("high", "medium"):
            overlap.append(rec)

    evidence = {
        "focus_window": {"start": str(start.date()), "end": str(end.date()),
                         "delta_vs_previous": round(drop, 2)},
        "decomposition_top_drivers": {
            d: rows[:4] for d, rows in decomp.items()
        },
        "anomalies_in_window": [
            {k: r.get(k) for k in ("entity", "date", "tier", "attribution")}
            for r in overlap[:8]
        ],
        "leakage_total_in_scope": (state.results.get("leakage") or {}
                                   ).get("total_impact_estimate"),
    }

    def _fallback() -> dict:
        hyps = []
        for d, rows in decomp.items():
            if rows and rows[0]["delta"] < 0:
                hyps.append({
                    "hypothesis": f"decline concentrated in {d}='{rows[0]['value']}' "
                                  f"(delta {rows[0]['delta']:,.0f})",
                    "confidence": 0.6,
                    "evidence": [f"decomposition.{d}"],
                })
        if overlap:
            hyps.append({
                "hypothesis": f"{len(overlap)} high/medium anomalies fall inside the "
                              "focus window and may explain part of the movement",
                "confidence": 0.5, "evidence": ["anomalies_in_window"],
            })
        return {"hypotheses": hyps[:5],
                "reasoning": "deterministic ranking of decomposition deltas"}

    if use_llm:
        llm = call_json(
            role="reasoner",
            system=("Think step by step: examine the decomposition evidence first, then form hypotheses ranked by that evidence. " + 
                "You are a revenue root-cause analyst. From ONLY the evidence "
                "given, produce ranked hypotheses for the revenue movement. Cite "
                "numbers from the evidence — do not invent any. Respond ONLY with "
                'JSON: {"hypotheses": [{"hypothesis": str, "confidence": float, '
                '"evidence": [str]}], "reasoning": str}'
            ),
            user=json.dumps(evidence, default=str),
            required_keys=("hypotheses",),
            fallback=_fallback,
        )
        parsed = llm.parsed
        agent = "deterministic" if llm.used_fallback else "llama-3.1-8b"
    else:
        parsed = _fallback()
        agent = "deterministic"

    out = {"evidence": evidence, "hypotheses": parsed.get("hypotheses", [])}
    state.results["rca"] = out
    state.ledger.log(
        stage="rca", agent=agent,
        decision=f"{len(out['hypotheses'])} ranked hypothesis(es) for "
                 f"{evidence['focus_window']['start']}→{evidence['focus_window']['end']} "
                 f"(delta {drop:,.0f})",
        reasoning=parsed.get("reasoning", "")
                  or "; ".join(h["hypothesis"] for h in out["hypotheses"][:2]),
        evidence=["rca.decomposition", "anomaly.flagged"],
        confidence=max((h.get("confidence", 0) for h in out["hypotheses"]), default=None),
    )
    return out
