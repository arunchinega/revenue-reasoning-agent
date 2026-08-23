"""
stages/domain_detect.py — Stage 2.0: Domain Detection + Context Assembly.

Context sources (priority order):
  1. data-reading outcome (always) — column names, sample values
  2. NL request (always)
  3. README/context doc (optional booster)

Heuristic (schema-signal scoring) runs first and is always available;
LLM refines/confirms when Ollama is up. Both paths ledger-logged.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from core.llm import call_json
from core.state import RunState

PROFILES_PATH = Path(__file__).resolve().parents[1] / "domain_profiles" / "profiles.yaml"
HITL_THRESHOLD = 0.6


def load_profiles() -> dict:
    return yaml.safe_load(PROFILES_PATH.read_text())


def assemble_context(state: RunState, max_sample_rows: int = 3) -> str:
    """The context block injected into every reasoning prompt."""
    cols = state.ingest_report["column_names"]
    sample = state.raw_df.head(max_sample_rows).to_dict(orient="records")
    parts = [
        f"COLUMNS: {cols}",
        f"SAMPLE ROWS: {sample}",
        f"ROWS: {state.ingest_report['rows']}",
        f"USER REQUEST: {state.nl_request or '(none provided)'}",
    ]
    if state.readme_text.strip():
        parts.append(f"CONTEXT DOC (user-provided):\n{state.readme_text[:2000]}")
    else:
        parts.append("CONTEXT DOC: none provided")
    return "\n".join(parts)


def _heuristic_domain(state: RunState, profiles: dict) -> dict:
    """Schema-signal scoring: fraction-weighted hits of signals in column names + request."""
    haystack = " ".join(state.ingest_report["column_names"]).lower()
    haystack += " " + state.nl_request.lower() + " " + state.readme_text[:1000].lower()
    best_name, best_score, best_hits = "generic", 0.0, []
    for name, prof in profiles.items():
        signals = prof.get("schema_signals") or []
        if not signals:
            continue
        hits = [s for s in signals if s in haystack]
        score = len(hits) / len(signals)
        if score > best_score:
            best_name, best_score, best_hits = name, score, hits
    confidence = min(0.5 + best_score, 0.95) if best_name != "generic" else 0.5
    return {
        "name": best_name if best_score >= 0.2 else "generic",
        "confidence": round(confidence if best_score >= 0.2 else 0.5, 2),
        "evidence": best_hits,
        "method": "heuristic",
    }


def run_domain_detection(state: RunState, use_llm: bool = True) -> RunState:
    profiles = load_profiles()
    heuristic = _heuristic_domain(state, profiles)
    result = heuristic

    if use_llm:
        domains = {k: v["display"] for k, v in profiles.items()}
        llm = call_json(
            role="reasoner",
            system=(
                "You classify the business domain of a dataset. "
                f"Choose exactly one of: {list(domains.keys())} ({domains}). "
                'Respond ONLY with JSON: {"domain": str, "confidence": float 0-1, '
                '"evidence": [strings citing specific columns/values/request words], '
                '"reasoning": str (one sentence)}'
            ),
            user=assemble_context(state),
            required_keys=("domain", "confidence"),
            fallback=lambda: heuristic | {"method": "heuristic_fallback"},
        )
        parsed = llm.parsed or {}
        name = parsed.get("domain", heuristic["name"])
        if name in profiles:
            result = {
                "name": name,
                "confidence": float(parsed.get("confidence", heuristic["confidence"])),
                "evidence": parsed.get("evidence", heuristic["evidence"]),
                "reasoning": parsed.get("reasoning", ""),
                "method": "heuristic_fallback" if llm.used_fallback else "llm",
            }

    result["profile"] = profiles[result["name"]]
    state.domain = result
    hitl = result["confidence"] < HITL_THRESHOLD
    state.ledger.log(
        stage="domain_detection",
        agent="deterministic" if result["method"].startswith("heuristic") else "llama-3.1-8b",
        decision=f"Domain: {result['name']} ({profiles[result['name']]['display']})",
        reasoning=result.get("reasoning")
                  or f"schema signals matched: {result['evidence'] or 'none — generic fallback'}",
        evidence=[f"ingest.column_names→{e}" for e in result["evidence"][:6]],
        confidence=result["confidence"],
        hitl_required=hitl,
        data={"leakage_rules_activated": result["profile"]["leakage_rules"]},
    )
    return state
