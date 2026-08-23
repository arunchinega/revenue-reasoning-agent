"""
stages/intent_detect.py — Stage 2: Intent Detection (LLM-first, domain-aware).

Canonical intents: leakage, forecasting, anomaly, rca, whatif, recommend, segment.
Llama 8B primary with domain vocab injected; keyword scorer is the fallback and
the latency shortcut for dead-obvious universal phrasing.
"""
from __future__ import annotations

from core.llm import MODELS, as_text, as_float, as_str_list, call_json
from core.state import RunState
from stages.domain_detect import assemble_context

INTENTS = ("leakage", "forecasting", "anomaly", "rca", "whatif", "recommend", "segment")

UNIVERSAL_KEYWORDS = {
    "forecasting": ["forecast", "predict", "projection", "next quarter", "next month", "next year"],
    "anomaly": ["anomal", "outlier", "unusual", "abnormal", "spike"],
    "leakage": ["leakage", "unbilled", "under-billed", "underbilled", "revenue loss", "slippage"],
    "rca": ["why", "root cause", "reason for", "driver", "what caused", "explain the"],
    "whatif": ["what if", "what-if", "scenario", "simulate", "if we increase", "if we decrease"],
    "recommend": ["recommend", "suggest", "what should", "action", "how do we improve"],
    "segment": ["segment", "cluster", "cohort", "customer group", "persona"],
}

HITL_THRESHOLD = 0.7
DEPENDENCIES = {  # auto-added upstream intents
    "rca": ["anomaly"],
    "whatif": ["forecasting"],
    "recommend": ["leakage", "forecasting"],
}


def _keyword_intents(request: str, domain_vocab: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Score request against universal + domain vocab. Returns (intents, hit-evidence)."""
    low = request.lower()
    found: dict[str, list[str]] = {}
    for intent, words in UNIVERSAL_KEYWORDS.items():
        hits = [w for w in words if w in low]
        if hits:
            found[intent] = hits
    for intent, words in (domain_vocab or {}).items():
        hits = [w for w in words if w.lower() in low]
        if hits:
            found.setdefault(intent, []).extend(hits)
    return list(found.keys()), found


def _resolve_dependencies(intents: list[str]) -> tuple[list[str], list[str]]:
    """Add upstream intents needed by requested ones; keep canonical exec order."""
    wanted = set(intents)
    added = []
    for intent in list(wanted):
        for dep in DEPENDENCIES.get(intent, []):
            if dep not in wanted:
                wanted.add(dep)
                added.append(f"{dep} (needed by {intent})")
    ordered = [i for i in ("segment", "anomaly", "leakage", "forecasting",
                           "rca", "whatif", "recommend") if i in wanted]
    return ordered, added


def run_intent_detection(state: RunState, use_llm: bool = True) -> RunState:
    request = state.nl_request.strip()
    vocab = (state.domain.get("profile") or {}).get("intent_vocab", {})
    kw_intents, kw_hits = _keyword_intents(request, vocab)

    result: dict = {}
    if not request:
        # no request at all → sensible default, flagged for HITL
        result = {
            "intents": ["anomaly", "forecasting"],
            "confidence": 0.5,
            "reasoning": "No request provided; defaulting to anomaly + forecast overview",
            "method": "default",
        }
    elif use_llm:
        vocab_note = "\n".join(
            f"  {i}: e.g. {', '.join(w[:4])}" for i, w in vocab.items()
        ) or "  (generic)"
        llm = call_json(
            role="reasoner",
            system=(
                "You map a user's analytics request to canonical intents. "
                f"Canonical intents: {list(INTENTS)}.\n"
                f"Domain-specific phrasings ({state.domain.get('name', 'generic')}):\n{vocab_note}\n"
                'Multiple intents are allowed. Respond ONLY with JSON: '
                '{"intents": [str], "confidence": float 0-1, '
                '"reasoning": str (one sentence, cite the request words that drove each intent)}'
            ),
            user=assemble_context(state),
            required_keys=("intents", "confidence"),
            fallback=lambda: {
                "intents": kw_intents or ["anomaly", "forecasting"],
                "confidence": 0.75 if kw_intents else 0.4,
                "reasoning": f"keyword fallback; hits: {kw_hits}" if kw_hits
                             else "keyword fallback found nothing; defaulting",
            },
        )
        parsed = llm.parsed or {}
        intents = [i for i in as_str_list(parsed.get("intents", [])) if i in INTENTS]
        # INTENT FLOOR: the LLM may sharpen the reading but never shrink it —
        # union with the deterministic keyword detector so an explicitly
        # requested capability can never be silently dropped
        missed = [i for i in kw_intents if i not in intents]
        if intents and missed:
            intents = intents + missed
        floor_note = (f" | intent floor restored: {missed} (keyword-evident "
                      f"in the request)" if intents and missed else "")
        result = {
            "intents": intents or kw_intents or ["anomaly", "forecasting"],
            "confidence": as_float(parsed.get("confidence"), 0.4),
            "reasoning": as_text(parsed.get("reasoning", "")) + floor_note,
            "method": "keyword_fallback" if llm.used_fallback else "llm",
        }
    else:
        result = {
            "intents": kw_intents or ["anomaly", "forecasting"],
            "confidence": 0.75 if kw_intents else 0.4,
            "reasoning": f"keyword hits: {kw_hits}",
            "method": "keyword",
        }

    ordered, dep_added = _resolve_dependencies(result["intents"])
    state.intents = ordered
    hitl = result["confidence"] < HITL_THRESHOLD

    state.ledger.log(
        stage="intent_detection",
        agent=MODELS["reasoner"] if result["method"] == "llm" else "deterministic",
        decision=f"Intents: {ordered}",
        reasoning=result["reasoning"]
                  + (f"; dependencies auto-added: {dep_added}" if dep_added else ""),
        evidence=[f"request→{h}" for hits in kw_hits.values() for h in hits[:2]]
                 or ["nl_request"],
        confidence=result["confidence"],
        hitl_required=hitl,
        data={"requested": result["intents"], "execution_order": ordered},
    )
    return state
