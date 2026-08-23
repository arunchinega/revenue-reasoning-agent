"""
agents/narrator.py — the Narration agent.

Turns the structured results in RunState into an executive summary.
Same contract as everything else: LLM (Gemma) sharpens the prose when
available; a deterministic template produces a complete, correct summary
when it isn't. The template IS the guarantee — the LLM is garnish.
"""
from __future__ import annotations

import json

from core.state import RunState


def _fmt(n: float) -> str:
    try:
        return f"{n:,.0f}"
    except Exception:  # noqa: BLE001
        return str(n)


def _deterministic_summary(state: RunState) -> str:
    """Narrative arc: headline → diagnosis → confidence → action."""
    r = state.results
    dom = state.domain.get("profile", {}).get("display", state.domain.get("name", "generic"))
    parts: list[str] = []

    # 1) HEADLINE — the single most material number
    lk = r.get("leakage") if "error" not in r.get("leakage", {}) else None
    fc = r.get("forecasting") if "error" not in r.get("forecasting", {}) else None
    if lk and lk.get("total_impact_estimate"):
        parts.append(f"### 💸 An estimated **{_fmt(lk['total_impact_estimate'])}** in "
                     f"revenue is leaking\n{lk.get('candidate_count', 0)} billing "
                     f"records across the {dom.lower()} data show recoverable revenue "
                     f"loss.")
    elif fc:
        parts.append(f"### 📈 Revenue forecast: `{fc.get('verdict')}` — "
                     f"{fc.get('winner_label')}")

    # 2) DIAGNOSIS — what is driving it
    diag: list[str] = []
    if lk:
        fired = {k: v for k, v in lk.get("rules_fired", {}).items() if v.get("hits")}
        if fired:
            top = sorted(fired.items(), key=lambda kv: -kv[1].get("impact_total", 0))[:3]
            diag.append("Leakage decomposes into "
                        + "; ".join(f"**{k}** ({_fmt(v['impact_total'])}, {v['hits']} hits)"
                                    for k, v in top) + ".")
    rc = r.get("rca") if "error" not in r.get("rca", {}) else None
    if rc and rc.get("hypotheses"):
        ev = rc.get("evidence", {}).get("focus_window", {})
        diag.append(f"The sharpest revenue movement was {ev.get('start', '?')} → "
                    f"{ev.get('end', '?')} (Δ {_fmt(ev.get('delta_vs_previous', 0))}); "
                    f"most likely cause: {rc['hypotheses'][0].get('hypothesis', '')} "
                    f"({rc['hypotheses'][0].get('confidence', 0):.0%} confidence).")
    an = r.get("anomaly") if "error" not in r.get("anomaly", {}) else None
    if an:
        c = an.get("counts", {})
        diag.append(f"{c.get('high', 0)} high-confidence anomalies corroborate "
                    f"({len(an.get('detectors_run', []))} detectors voting; "
                    f"{c.get('medium', 0)} medium, {c.get('review', 0)} parked for review).")
    if diag:
        parts.append("**Why:** " + " ".join(diag))

    # 3) CONFIDENCE — forecast verdict in plain language, escalation included
    if fc:
        v = fc.get("verdict")
        if v == "accept_ensemble" and fc.get("ensemble"):
            e = fc["ensemble"]
            conf = (f"I certify the forward view with an **ensemble** of "
                    f"{' + '.join(e['members'])} (blended backtest MAPE "
                    f"{e['blend_mape']:.2f}%) — their errors were complementary, "
                    f"so the blend beats any single model.")
        elif v == "accept_single":
            best = next((m for m in fc.get("metrics", [])
                         if m["model"] == fc.get("winner")), None)
            conf = (f"I certify the forward view on **{fc.get('winner')}** "
                    + (f"(backtest MAPE {best['mape']:.2f}%)" if best and
                       best.get("mape") is not None else "")
                    + f", selected from {len([m for m in fc.get('metrics', []) if not m.get('error')])} "
                      f"competing models in {fc.get('attempts', 1)} attempt(s).")
        else:
            conf = (f"⚠️ I tested {len([m for m in fc.get('metrics', []) if not m.get('error')])} "
                    f"models and **declined to certify a forecast**: "
                    f"{fc.get('critic_reasoning', 'no candidate met the quality bar')}. "
                    f"The best attempt ({fc.get('best_available', '—')}) is shown "
                    f"with that caveat — refusing to overstate confidence is the "
                    f"safeguard working.")
        parts.append("**Forecast confidence:** " + conf)

    sg = r.get("segment") if "error" not in r.get("segment", {}) else None
    if sg:
        personas = ", ".join(p.get("persona", "?") for p in sg.get("profiles", [])[:3])
        parts.append(f"**Customer base:** {sg.get('k')} segments "
                     f"(silhouette {sg.get('silhouette', 0):.2f}) — {personas}.")

    # 4) ACTION — what to do, traceable
    rec = r.get("recommend") if "error" not in r.get("recommend", {}) else None
    if rec and rec.get("recommendations"):
        parts.append("**Do next:**")
        for x in rec["recommendations"][:3]:
            parts.append(f"1. {x.get('action')} — *effort {x.get('effort', '?')}, "
                         f"confidence {x.get('confidence', 0):.0%}, evidence "
                         f"`{x.get('traces_to', '?')}`*")

    failed = [k for k, v in r.items() if isinstance(v, dict) and "error" in v]
    if failed:
        parts.append(f"*Capabilities that failed this run: {', '.join(failed)}.*")
    return "\n\n".join(parts)


def _evidence_digest(state: RunState) -> str:
    """Compact JSON evidence pack for the LLM narrator (~2-3K tokens)."""
    r = state.results
    digest = {
        "domain": state.domain.get("name"),
        "rows": state.ingest_report.get("rows"),
        "nl_request": state.nl_request,
    }
    if "leakage" in r:
        digest["leakage"] = {k: r["leakage"].get(k) for k in
                             ("candidate_count", "total_impact_estimate")}
    if "anomaly" in r:
        digest["anomaly_counts"] = r["anomaly"].get("counts")
    if "forecasting" in r:
        fc = r["forecasting"]
        digest["forecasting"] = {"verdict": fc.get("verdict"), "winner": fc.get("winner"),
                                 "metrics": fc.get("metrics")}
    if "rca" in r:
        digest["rca_hypotheses"] = r["rca"].get("hypotheses")
    if "segment" in r:
        digest["segments"] = r["segment"].get("profiles")
    if "whatif" in r:
        digest["whatif"] = r["whatif"].get("scenarios")
    if "recommend" in r:
        digest["recommendations"] = r["recommend"].get("recommendations")
    return json.dumps(digest, default=str)


NUM_RE = __import__("re").compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers_in(text: str) -> set[str]:
    """Normalized numeric tokens (commas stripped, trailing zeros trimmed)."""
    out = set()
    for tok in NUM_RE.findall(text or ""):
        t = tok.replace(",", "")
        try:
            f = float(t)
        except ValueError:
            continue
        out.add(f"{f:g}")
    return out


def _llm_numbers_check(llm_text: str, evidence_json: str, template: str) -> bool:
    """Every number the LLM wrote must exist in the evidence digest or the
    deterministic template (allowing rounding to fewer decimals). Guards
    against distorted claims like leakage restated as a revenue decline with
    an invented currency."""
    allowed = _numbers_in(evidence_json) | _numbers_in(template)
    # allow common rounded forms of every allowed number
    expanded = set(allowed)
    for a in allowed:
        try:
            f = float(a)
        except ValueError:
            continue
        expanded.update({f"{round(f):g}", f"{round(f, 1):g}", f"{round(f, 2):g}"})
    hallucinated = [n for n in _numbers_in(llm_text)
                    if n not in expanded and float(n) > 12]  # small ints = list numbering etc.
    return len(hallucinated) == 0


def run_narration(state: RunState, use_llm: bool = True) -> str:
    """Produce the executive summary; log the decision to the ledger.

    HYBRID CONTRACT: the deterministic template is always computed and its
    HEADLINE + numbers are ground truth. The LLM may only produce prose whose
    every number exists in the evidence; otherwise we keep the template."""
    template = _deterministic_summary(state)

    if use_llm:
        from core.llm import call_text
        res = call_text(
            role="narrator",
            system=("You are the narration agent of a revenue reasoning system. "
                    "Write a crisp executive summary (150-250 words) for a revenue "
                    "operations leader following this exact arc: (1) HEADLINE — the "
                    "single most material number; (2) WHY — the drivers; "
                    "(3) CONFIDENCE — the forecast verdict in plain language; if the "
                    "verdict is escalate, present the refusal to certify as rigor, "
                    "not failure; (4) DO NEXT — top actions with their evidence. "
                    "Use ONLY facts in the evidence JSON — never invent numbers, and ""NEVER attach a currency symbol ($, €, ₹) — figures are currency-agnostic. Leakage is recoverable billing loss, NOT a revenue-decline forecast — never conflate the two. "
                    "Keep every figure exactly as given."),
            user=_evidence_digest(state),
            fallback_text=template,
        )
        digest = _evidence_digest(state)
        llm_text = (res.content or "").strip()
        if res.used_fallback or not llm_text:
            summary, agent = template, "deterministic"
            reasoning = f"LLM unavailable ({res.error}); template summary used"
        elif not _llm_numbers_check(llm_text, digest, template):
            summary, agent = template, "deterministic"
            reasoning = ("LLM narration REJECTED by number-guard: it contained "
                         "figures not present in the evidence digest — "
                         "deterministic template used instead")
        else:
            summary = llm_text
            agent = "gemma-2-2b"
            reasoning = "LLM narration over evidence digest (passed number-guard)"
    else:
        summary, agent = template, "deterministic"
        reasoning = "LLM disabled; template summary assembled from results"

    state.ledger.log(stage="narration", agent=agent,
                     decision="Executive summary generated",
                     reasoning=reasoning,
                     evidence=[f"results.{k}" for k in state.results])
    state.results["narration"] = {"summary": summary}
    return summary


def build_report_markdown(state: RunState, summary: str) -> str:
    """Full exportable report: summary + per-capability detail + ledger."""
    parts = [f"# Revenue Reasoning Agent — Run Report",
             f"*Run {state.run_id} · domain: {state.domain.get('name', '?')} · "
             f"request: \"{state.nl_request}\"*", "",
             "## Executive Summary", summary, ""]
    r = state.results
    if "forecasting" in r and r["forecasting"].get("metrics"):
        fc = r["forecasting"]
        parts += [f"## Forecast bake-off — winner: {fc.get('winner_label', fc.get('winner'))}", "",
                  "| model | family | MAPE % | RMSE | folds |", "|---|---|---|---|---|"]
        for m in sorted(r["forecasting"]["metrics"],
                        key=lambda x: x.get("mape") if x.get("mape") is not None else 1e9):
            if m.get("error"):
                continue
            mape = f"{m['mape']:.2f}" if m.get("mape") is not None else "—"
            rmse = f"{m['rmse']:,.0f}" if m.get("rmse") is not None else "—"
            folds = m.get("folds") if m.get("folds") is not None else "—"
            parts.append(f"| {m['model']} | {m['family']} | {mape} | {rmse} | {folds} |")
        parts.append("")
    if "leakage" in r and r["leakage"].get("candidates"):
        parts += ["## Top leakage candidates", "",
                  "| entity | date | rules | impact |", "|---|---|---|---|"]
        for c in r["leakage"]["candidates"][:10]:
            parts.append(f"| {c.get('entity', '?')} | {c.get('date', '?')} | "
                         f"{', '.join(c.get('rules', []))} | {c.get('impact_estimate', 0):,.0f} |")
        parts.append("")
    parts += ["---", "", state.ledger.to_markdown()]
    return "\n".join(parts)
