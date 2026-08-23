"""
stages/column_map.py — Stage 2.5: Column Mapping (heuristic → LLM fallback).

Maps roles {date_column, target_column, id_column, feature_columns} with a
confidence score; low confidence → HITL confirm panel (HITL point #1).
"""
from __future__ import annotations

from core.llm import MODELS, as_text, as_float, as_str_list, call_json
from core.state import RunState
from stages.domain_detect import assemble_context

HITL_THRESHOLD = 0.8

TARGET_HINTS = ("revenue", "billed", "amount", "sales", "total", "value", "charge", "premium", "fee")
ID_HINTS = ("customer", "account", "client", "meter", "policy", "member", "user_id")


def _heuristic_map(state: RunState) -> dict:
    ing = state.ingest_report
    eda = state.eda_report
    cols = ing["column_names"]
    date_col = ing["date_columns"][0] if ing["date_columns"] else None
    target = eda.get("target_guess")

    id_col = None
    for hint in ID_HINTS:
        for c in cols:
            if hint in c.lower():
                id_col = c
                break
        if id_col:
            break

    features = [
        c for c in ing["numeric_columns"]
        if c not in (target, id_col) and state.raw_df[c].nunique() > 1
    ]
    # confidence: strong if target matched a semantic hint, weaker if variance fallback
    target_hint_hit = target and any(h in target.lower() for h in TARGET_HINTS)
    confidence = 0.9 if (target_hint_hit and date_col) else 0.7 if target_hint_hit else 0.55
    return {
        "date_column": date_col,
        "target_column": target,
        "id_column": id_col,
        "feature_columns": features,
        "confidence": confidence,
        "reasoning": (
            f"target '{target}' matched semantic hint" if target_hint_hit
            else f"target '{target}' chosen by variance fallback"
        ),
        "method": "heuristic",
    }


def run_column_mapping(state: RunState, use_llm: bool = True) -> RunState:
    heur = _heuristic_map(state)
    result = heur

    # LLM refinement only when the heuristic is unsure
    if use_llm and heur["confidence"] < 0.9:
        llm = call_json(
            role="reasoner",
            system=(
                "You map dataset columns to analytical roles. Roles: date_column "
                "(time axis), target_column (the revenue/value to analyze), id_column "
                "(customer/account identifier or null), feature_columns (numeric drivers). "
                'Respond ONLY with JSON: {"date_column": str|null, "target_column": str, '
                '"id_column": str|null, "feature_columns": [str], "confidence": float 0-1, '
                '"reasoning": str}. Column names must come from COLUMNS exactly.'
            ),
            user=assemble_context(state)
                 + f"\nHEURISTIC GUESS (refine or confirm): {heur}",
            required_keys=("target_column", "confidence"),
            fallback=lambda: heur | {"method": "heuristic_fallback"},
        )
        parsed = llm.parsed or {}
        cols = set(state.ingest_report["column_names"])
        if parsed.get("target_column") in cols:
            result = {
                "date_column": parsed.get("date_column") if parsed.get("date_column") in cols else heur["date_column"],
                "target_column": parsed["target_column"],
                "id_column": parsed.get("id_column") if parsed.get("id_column") in cols else heur["id_column"],
                "feature_columns": [c for c in parsed.get("feature_columns", []) if c in cols] or heur["feature_columns"],
                "confidence": as_float(parsed.get("confidence"), heur["confidence"]),
                "reasoning": as_text(parsed.get("reasoning", heur["reasoning"])),
                "method": "heuristic_fallback" if llm.used_fallback else "llm",
            }

    state.column_map = result
    hitl = result["confidence"] < HITL_THRESHOLD
    state.ledger.log(
        stage="column_mapping",
        agent="deterministic" if result["method"].startswith("heuristic") else MODELS["reasoner"],
        decision=(
            f"date={result['date_column']}, target={result['target_column']}, "
            f"id={result['id_column']}, {len(result['feature_columns'])} feature col(s)"
        ),
        reasoning=result["reasoning"],
        evidence=["eda.target_guess", "ingest.date_columns"],
        confidence=result["confidence"],
        hitl_required=hitl,
    )
    return state


def apply_hitl_correction(state: RunState, corrected: dict) -> RunState:
    """Called by UI when the human edits the mapping (HITL point #1)."""
    state.column_map.update(corrected)
    state.column_map["confidence"] = 1.0
    state.ledger.log(
        stage="column_mapping",
        agent="human",
        decision=f"Human corrected mapping: {corrected}",
        reasoning="HITL confirmation panel",
        confidence=1.0,
        hitl_required=True,
        hitl_resolution="corrected by user",
    )
    return state
