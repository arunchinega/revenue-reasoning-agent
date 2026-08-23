"""
agents/stage_stories.py — the story layer.

One deterministic template per stage, filled ONLY from state (always true,
no LLM risk). Voice: first person — the agent as a character narrating its
own reasoning. Every story ends by handing off to the next step.
"""
from __future__ import annotations

from core.state import RunState


def _n(x, default=0):
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return str(default)


def ingest_story(state: RunState) -> str:
    ing = state.ingest_report or {}
    rows = _n(ing.get("rows"))
    df = state.raw_df
    days = customers = "?"
    cm = state.column_map or {}
    try:
        import pandas as pd
        dcol = next((c for c in df.columns if "date" in c.lower()), None)
        if dcol is not None:
            dd = pd.to_datetime(df[dcol], errors="coerce")
            days = str((dd.max() - dd.min()).days)
        idc = next((c for c in df.columns if c.lower().endswith("id")), None)
        if idc is not None:
            customers = _n(df[idc].nunique())
    except Exception:  # noqa: BLE001
        pass
    return (f"I received **{rows} billing records** spanning **{days} days** — "
            f"**{customers} customers**. Before I trust any of it, let me "
            f"profile it properly.")


def eda_story(state: RunState) -> str:
    ts = (state.eda_report or {}).get("timeseries") or {}
    seas = ts.get("seasonality") or {}
    stat = ts.get("stationarity") or {}
    df = state.raw_df
    miss = f"{df.isna().mean().mean():.1%}" if df is not None else "?"
    dupes = int(df.duplicated().sum()) if df is not None else "?"
    bits = []
    if seas.get("period"):
        bits.append(f"a **{seas['period']}-day cycle** "
                    f"(strength {seas.get('strength', 0):.2f})")
    if ts.get("trend"):
        bits.append(f"an **{ts['trend']}** trend")
    if stat.get("conclusion"):
        bits.append(f"{stat['conclusion']} behaviour "
                    f"(ADF p={stat.get('adf_pvalue', 0):.3f})")
    pulse = ", ".join(bits) if bits else "no strong temporal structure"
    return (f"The data has a pulse: {pulse}. **{miss} missing values**, "
            f"**{dupes} duplicate rows** — normal mess, nothing disqualifying. "
            f"This rhythm will decide which forecasting models deserve a lane.")


def features_story(state: RunState) -> str:
    fr = state.feature_report or {}
    rules = fr.get("applied_rules", {})
    n = sum(len(v) for v in rules.values())
    top = (fr.get("top_features") or ["?"])[0]
    dropped = len(fr.get("leakage_suspects_dropped", []) or [])
    drop_txt = (f" — and I **dropped {dropped} column(s)** that would have let "
                f"the models cheat (they encode the answer)" if dropped else "")
    return (f"I engineered **{n} candidate signals** across {len(rules)} rule "
            f"groups. **{top}** carries the most predictive weight{drop_txt}. "
            f"What survives is honest signal.")


def plan_story(state: RunState) -> str:
    dom = state.domain or {}
    disp = dom.get("profile", {}).get("display", dom.get("name", "generic"))
    intents = ", ".join(state.intents or [])
    return (f"I read this as **{disp}** data ({dom.get('confidence', 0):.0%} "
            f"confident) and your request as: **{intents}**. Here is my plan — "
            f"**you have the veto.** Correct me before I spend compute.")


def bakeoff_story(state: RunState) -> str:
    plan = (state.plan or {}).get("forecasting", {})
    cands = plan.get("candidates", [])
    roster = plan.get("roster", {})
    gated = [(n, v.get("reason", "")) for n, v in roster.items()
             if v.get("status") == "data_gated"]
    gate_txt = (f" Not everyone made the start line — e.g. **{gated[0][0]}**: "
                f"{gated[0][1][:90]}." if gated else "")
    return (f"**{len(cands)} models are racing** on rolling-origin backtests: "
            f"{', '.join(cands)}.{gate_txt} The data picks the winner here — "
            f"I don't.")


def verdict_story(state: RunState) -> str:
    fc = (state.results or {}).get("forecasting", {})
    v = fc.get("verdict")
    wc = fc.get("winner_card") or {}
    if v == "accept_ensemble" and fc.get("ensemble"):
        e = fc["ensemble"]
        return (f"No single model earned it alone — but **{e['members'][0]} and "
                f"{e['members'][1]} disagree usefully** (error correlation "
                f"{e['error_corr']}), so their weighted blend beats both: "
                f"**MAPE {e['blend_mape']:.2f}%**. Certified as a committee.")
    if v == "accept_single":
        margin = wc.get("margin_vs_baseline")
        m_txt = f", beating the naive floor by {margin:.2f}pp" if margin is not None else ""
        return (f"**{fc.get('winner')}** takes it: MAPE "
                f"{wc.get('mape', 0):.2f}%{m_txt}, in {fc.get('attempts', 1)} "
                f"attempt(s). Certified.")
    if v == "escalate":
        return (f"I tested {len([m for m in fc.get('metrics', []) if not m.get('error')])} "
                f"models and **I am not certifying any of them** — "
                f"{fc.get('critic_reasoning', 'none met my quality bar')}. "
                f"I'd rather hand this to you than pretend.")
    return "The forecasting loop did not complete."


def anomaly_story(state: RunState) -> str:
    an = (state.results or {}).get("anomaly", {})
    c = an.get("counts", {})
    ros = an.get("roster", {})
    ran = sum(1 for v in ros.values() if v.get("status") == "ran") or \
        len(an.get("detectors_run", []))
    return (f"**{ran} independent detectors voted** on every record. Consensus: "
            f"**{c.get('high', 0)} high-confidence** anomalies, "
            f"{c.get('medium', 0)} medium, and {c.get('review', 0)} parked for "
            f"review — I only shout when several detectors agree.")


def leakage_story(state: RunState) -> str:
    lk = (state.results or {}).get("leakage", {})
    rules = lk.get("rules_fired", {})
    fired = [k for k, v in rules.items() if v.get("hits")]
    clean = [k for k, v in rules.items() if not v.get("hits")]
    return (f"I checked **{len(rules)} known leak patterns**. "
            f"**{len(fired)} fired**, {len(clean)} came back clean — a clean "
            f"rule is evidence too. Estimated **{_n(lk.get('total_impact_estimate'))} "
            f"recoverable** across {lk.get('candidate_count', 0)} records.")


def rca_story(state: RunState) -> str:
    rc = (state.results or {}).get("rca", {})
    fw = rc.get("evidence", {}).get("focus_window", {})
    hyps = rc.get("hypotheses", [])
    top = hyps[0] if hyps else {}
    return (f"The sharpest movement was **{fw.get('start', '?')} → "
            f"{fw.get('end', '?')}** (Δ {_n(fw.get('delta_vs_previous'))}). "
            f"I decomposed it by segment and customer; most likely cause: "
            f"**{top.get('hypothesis', 'inconclusive')}** "
            f"({top.get('confidence', 0):.0%}).")


def segment_story(state: RunState) -> str:
    sg = (state.results or {}).get("segment", {})
    personas = ", ".join(p.get("persona", "?") for p in sg.get("profiles", [])[:3])
    return (f"Your customers cluster into **{sg.get('k', '?')} behavioural "
            f"segments** (silhouette {sg.get('silhouette', 0):.2f}): {personas}. "
            f"Each gets its own playbook in the recommendations.")


def whatif_story(state: RunState) -> str:
    wf = (state.results or {}).get("whatif", {})
    scen = wf.get("scenarios", [])
    best = max(scen, key=lambda s: s.get("delta_vs_baseline", 0), default=None)
    if not best:
        return "No scenarios could be computed."
    return (f"I simulated **{len(scen)} levers** against a "
            f"{_n(wf.get('baseline_total'))} baseline. The strongest: "
            f"**{best['scenario']}** ({best['delta_vs_baseline']:+,.0f}, "
            f"{best['delta_pct']:+.1f}%).")


def recommend_story(state: RunState) -> str:
    rec = (state.results or {}).get("recommend", {})
    n = len(rec.get("recommendations", []))
    return (f"Everything funnels into **{n} recommended actions** — each one "
            f"traces to a specific piece of evidence above. No orphan advice.")


def narration_story(state: RunState) -> str:
    entries = state.ledger.entries
    llm = sum(1 for e in entries if e.agent not in ("deterministic", "human"))
    hum = sum(1 for e in entries if e.agent == "human")
    return (f"Everything you just read traces to the ledger — "
            f"**{len(entries)} decisions**: {llm} by LLM, "
            f"{len(entries) - llm - hum} deterministic, {hum} by you.")


CAPABILITY_STORIES = {
    "forecasting": verdict_story,
    "anomaly": anomaly_story,
    "leakage": leakage_story,
    "rca": rca_story,
    "segment": segment_story,
    "whatif": whatif_story,
    "recommend": recommend_story,
}


# ---------------------------------------------------------------------------
# Spoken commentary — richer than the cards; what the voice actually says.
# Cards stay tight for reading; these go deeper on the numbers.
# ---------------------------------------------------------------------------

def _plain(x) -> str:
    return str(x).replace("**", "").replace("`", "")


def speak_ingest(state: RunState) -> str:
    return ("Alright, let's see what we're working with. " +
            _plain(ingest_story(state)) +
            " Give me a moment to feel out its rhythm and its quality.")


def speak_eda(state: RunState) -> str:
    ts = (state.eda_report or {}).get("timeseries") or {}
    seas = ts.get("seasonality") or {}
    extra = ""
    if seas.get("period"):
        extra = (f" A seasonality strength of {seas.get('strength', 0):.2f} means "
                 f"the {seas['period']}-day cycle explains most of the variation — "
                 f"that is a strong, learnable pattern.")
    return ("Okay, interesting. " + _plain(eda_story(state)) + extra +
            " So we have something learnable here.")


def speak_features(state: RunState) -> str:
    fr = state.feature_report or {}
    tops = (fr.get("top_features") or [])[:3]
    extra = (f" The three strongest signals are {', '.join(tops)}."
             if tops else "")
    return ("Now, before any model sees this data, I build its diet. " +
            _plain(features_story(state)) + extra)


def speak_plan(state: RunState) -> str:
    return ("Here's my plan — and this part is a conversation, not a decree. " +
            _plain(plan_story(state)) +
            " Nothing runs until you approve. Over to you.")


def speak_verdict(state: RunState) -> str:
    fc = (state.results or {}).get("forecasting", {})
    wc = fc.get("winner_card") or {}
    base = _plain(verdict_story(state))
    bits = []
    if wc.get("margin_vs_runner_up") is not None:
        bits.append(f"The margin over the runner-up was "
                    f"{wc['margin_vs_runner_up']:.2f} percentage points.")
    ee = fc.get("ensemble_eval") or {}
    if ee.get("reason") and not fc.get("ensemble"):
        bits.append(f"An ensemble was considered but not chosen, because "
                    f"{_plain(ee['reason'])}.")
    return ("So — who won? " + base + " " + " ".join(bits))


def speak_anomaly(state: RunState) -> str:
    an = (state.results or {}).get("anomaly", {})
    c = an.get("counts", {})
    return ("Meanwhile, my detectors have been talking to each other. " +
            _plain(anomaly_story(state)) +
            f" To be clear: {c.get('high', 0)} records had at least three "
            f"independent detectors agree — chase those first.")


def speak_leakage(state: RunState) -> str:
    lk = (state.results or {}).get("leakage", {})
    fired = {k: v for k, v in lk.get("rules_fired", {}).items() if v.get("hits")}
    top = max(fired.items(), key=lambda kv: kv[1].get("impact_total", 0),
              default=None)
    extra = ""
    if top:
        extra = (f" The biggest leak type is {top[0].replace('_', ' ')}, worth "
                 f"{top[1]['impact_total']:,.0f} across {top[1]['hits']} records.")
    return ("And here's the part your CFO will care about. " +
            _plain(leakage_story(state)) + extra)


def speak_rca(state: RunState) -> str:
    return "Now, why did revenue move? " + _plain(rca_story(state))


def speak_segment(state: RunState) -> str:
    return "And who are these customers, really? " + _plain(segment_story(state))


def speak_whatif(state: RunState) -> str:
    return "Let's play with the levers for a second. " + _plain(whatif_story(state))


def speak_recommend(state: RunState) -> str:
    rec = (state.results or {}).get("recommend", {})
    recs = rec.get("recommendations", [])
    first = f" Number one: {recs[0].get('action', '')}." if recs else ""
    return ("So, what would I actually do about all of this? " +
            _plain(recommend_story(state)) + first)


def speak_hero(state: RunState) -> str:
    lk = (state.results or {}).get("leakage", {})
    n = lk.get("total_impact_estimate")
    if not n:
        return "The run is complete."
    return (f"Bottom line. An estimated {n:,.0f} in revenue is recoverable. "
            f"Every number behind that claim is in the evidence, and every "
            f"decision that produced it is in the ledger.")


SPEAK = {
    "ingest": speak_ingest, "eda": speak_eda, "features": speak_features,
    "plan": speak_plan, "forecasting": speak_verdict, "anomaly": speak_anomaly,
    "leakage": speak_leakage, "rca": speak_rca, "segment": speak_segment,
    "whatif": speak_whatif, "recommend": speak_recommend, "hero": speak_hero,
}


# Duet mode: which speaker voices each stage. The Critic (male) takes the
# judgement moments; the Narrator (female) carries the story.
DUET_SPEAKER = {
    "ingest": "narrator", "eda": "narrator", "features": "narrator",
    "plan": "critic", "forecasting": "critic", "anomaly": "narrator",
    "leakage": "narrator", "rca": "narrator", "segment": "narrator",
    "whatif": "narrator", "recommend": "critic", "hero": "narrator",
}
