"""
app.py — Revenue Reasoning Agent UI, v17 multipage.

Five screens, one story:
  ▶ Run        — upload → plan approval → live reasoning ledger
  📖 Findings  — the story: headline, diagnosis, confidence, actions
  🔬 Evidence  — the proof: every chart, sectioned and collapsible
  🧠 Reasoning — the differentiator: full decision ledger with badges
  📄 Export    — report + raw ledger downloads

Visual identity: dark "revenue forensics"; three meaning-bearing hues used
identically everywhere — amber = money/leakage, teal = forecast/future,
red = anomaly/risk; slate for history and neutrals.
"""
from __future__ import annotations

import core.quiet  # noqa: F401  (must precede TF imports)

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from core.orchestrator import run_perception, ollama_available
from core.dispatcher import run_capabilities
from agents.narrator import run_narration, build_report_markdown

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    HAS_PLOTLY = False

st.set_page_config(page_title="Revenue Reasoning Agent", layout="wide",
                   page_icon="🧭")

INK, SURFACE, GRID = "#0E1418", "#182028", "#2A3540"
TEXT, SLATE = "#E8E6E0", "#8B98A9"
AMBER, TEAL, RED = "#E8A33D", "#2FBFA7", "#E85D4A"
TIER_COLORS = {"high": RED, "medium": AMBER, "review": SLATE}

INTENT_LABELS = {
    "segment": "Customer segmentation", "anomaly": "Anomaly detection",
    "leakage": "Revenue leakage", "forecasting": "Forecasting",
    "rca": "Root-cause analysis", "whatif": "What-if scenarios",
    "recommend": "Recommendations",
}
AGENT_BADGE = {"deterministic": "⚙️", "human": "🧑"}
SCREENS = ["▶ Run", "📖 Findings", "📊 Data & Features", "🔬 Evidence", "🧠 Reasoning", "📄 Export"]


# ---------------------------------------------------------------- helpers
def _badge(agent: str) -> str:
    return AGENT_BADGE.get(agent, "🤖") + " " + agent


def _entry_md(e) -> str:
    conf = f" · conf {e.confidence:.2f}" if e.confidence is not None else ""
    md = f"**`{e.stage}`** {_badge(e.agent)}{conf} — {e.decision}"
    if e.reasoning:
        md += f"\n\n> {e.reasoning[:400]}"
    if e.evidence:
        md += f"\n\n<sub>evidence: {', '.join(e.evidence[:6])}</sub>"
    return md


def _layout(fig, height=380, **kw):
    fig.update_layout(
        template=None, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=13),
        margin=dict(l=10, r=10, t=34, b=10),
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT, bordercolor=GRID),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def _show(fig):
    st.plotly_chart(fig, width='stretch',
                    config={"displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def _stream_ledger(state, container, already: int = 0) -> None:
    orig = state.ledger.log

    def hooked(**kw):
        entry = orig(**kw)
        with container:
            st.markdown(_entry_md(entry), unsafe_allow_html=True)
            st.divider()
        return entry

    state.ledger.log = hooked
    with container:
        for e in state.ledger.entries[already:]:
            st.markdown(_entry_md(e), unsafe_allow_html=True)
            st.divider()


def _frame(state) -> pd.DataFrame:
    return state.feature_df if state.feature_df is not None else state.raw_df


def _daily_series(state) -> pd.Series | None:
    cm = state.column_map
    try:
        df = _frame(state)[[cm["date_column"], cm["target_column"]]].copy()
        df[cm["date_column"]] = pd.to_datetime(df[cm["date_column"]])
        return df.groupby(cm["date_column"])[cm["target_column"]].sum().sort_index()
    except Exception:  # noqa: BLE001
        return None


FEMALE_HINTS = ["neerja", "heera", "swara", "zira", "aria", "jenny", "sonia",
                "natasha", "clara", "emma", "ava", "michelle", "libby", "female"]
MALE_HINTS = ["ravi", "prabhat", "madhur", "david", "mark", "guy", "ryan",
              "andrew", "brian", "thomas", "male"]


def _speak(text: str, speaker: str | None = None) -> None:
    """Browser TTS via SpeechSynthesis — no installs, works offline.
    Voice preference: Female (default) / Male / Auto. Utterances queue in
    the tab's shared synthesis engine, so commentary plays in card order
    even while cards keep appearing."""
    if not st.session_state.get("voice_on"):
        return
    import json as _json
    import streamlit.components.v1 as components
    pref = st.session_state.get("voice_pref", "Female")
    if pref == "Duet" and speaker:
        hints = MALE_HINTS if speaker == "critic" else FEMALE_HINTS
    else:
        hints = (FEMALE_HINTS if pref in ("Female", "Duet")
                 else MALE_HINTS if pref == "Male" else [])
    clean = (text.replace("**", "").replace("`", "").replace("#", "")
             .replace("→", " to ").replace("Δ", "delta "))
    components.html(
        "<script>(function(){try{"
        f"const HINTS={_json.dumps(hints)};"
        "function pick(){const vs=window.speechSynthesis.getVoices();"
        "if(!vs.length)return null;"
        "const en=vs.filter(v=>v.lang&&v.lang.toLowerCase().startsWith('en'));"
        "const pool=en.length?en:vs;"
        "for(const h of HINTS){const m=pool.find(v=>v.name.toLowerCase()"
        ".includes(h));if(m)return m;}return pool[0];}"
        f"const u=new SpeechSynthesisUtterance({_json.dumps(clean)});"
        "u.rate=1.0;u.pitch=1.0;"
        "const go=()=>{const v=pick();if(v)u.voice=v;"
        "window.speechSynthesis.speak(u);};"
        "if(window.speechSynthesis.getVoices().length)go();"
        "else window.speechSynthesis.onvoiceschanged=()=>{go();"
        "window.speechSynthesis.onvoiceschanged=null;};"
        "}catch(e){}})();</script>", height=0)


def _story_card(icon: str, title: str, story: str, speak: bool = True,
                script: str | None = None, stage_key: str | None = None) -> None:
    """Card shows the tight story; the voice speaks the richer script.
    In Duet mode the stage decides which of the two voices speaks."""
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.markdown(story)
    if speak:
        speaker = None
        if stage_key:
            from agents.stage_stories import DUET_SPEAKER
            speaker = DUET_SPEAKER.get(stage_key)
        _speak(script or f"{title}. {story}", speaker=speaker)


def _phase() -> str:
    return st.session_state.get("phase", "idle")


def _start_worker(target, *args) -> None:
    import threading
    t = threading.Thread(target=target, args=args, daemon=True)
    try:  # make the thread's session_state writes visible to the page (1.5x fix)
        from streamlit.runtime.scriptrunner import (add_script_run_ctx,
                                                    get_script_run_ctx)
        add_script_run_ctx(t, get_script_run_ctx())
    except Exception:  # noqa: BLE001
        pass
    st.session_state["worker"] = t
    st.session_state["worker_error"] = None
    t.start()


def _worker_alive() -> bool:
    t = st.session_state.get("worker")
    return bool(t is not None and t.is_alive())


def _need_results() -> bool:
    state = st.session_state.get("state")
    if _phase() != "done" or state is None:
        st.info("No completed run yet — go to **▶ Run**, upload a CSV and run "
                "the analysis. Results will light this screen up.")
        return True
    return False


# ================================================================ RUN SCREEN
def screen_run(up, readme_up, nl, use_llm):
    if _phase() == "idle":
        st.markdown(
            f"<div style='text-align:center;padding:60px 0 6px 0'>"
            f"<div style='font-size:56px'>🧭</div>"
            f"<div style='font-size:40px;font-weight:800;color:{TEXT}'>"
            f"What's happening in your revenue?</div>"
            f"<div style='font-size:17px;color:{SLATE};padding-top:8px'>"
            f"Upload billing data, ask in plain language — the agent reasons "
            f"out loud, shows its evidence, and signs its ledger.</div></div>",
            unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("&nbsp;")
            with st.container(border=True):
                st.markdown("**1** · Drop your CSV in the sidebar")
                st.markdown("**2** · Say what you want to know "
                            "(leakage? forecast? why revenue moved?)")
                st.markdown("**3** · Hit **▶ Run analysis** and watch it think "
                            "— 🔊 turn on the narrator for the full tour")
        return

    if _phase() == "perceive" and use_llm and not st.session_state.get("warmed"):
        st.info("⏳ Starting the run — first the local models load into memory "
                "(one-time, can take a minute on a cold start). If this page "
                "ever looks stuck, just refresh the browser tab — the run "
                "continues server-side.")
        with st.status("🔥 Warming up local models (one-time)…", expanded=True) as ws:
            from core.llm import warm_up
            for role, t in warm_up().items():
                ws.write(f"{role}: {'ready in ' + str(t) + 's' if t >= 0 else 'unreachable — will fall back'}")
            ws.update(label="Models warm", state="complete", expanded=False)
        st.session_state["warmed"] = True

    if _phase() == "perceive":
        if up is None and "pending_csv" not in st.session_state:
            st.session_state["phase"] = "idle"
            st.info("The uploaded file is no longer available — please upload "
                    "it again and press ▶ Run analysis.")
            st.stop()
        st.subheader("🔎 Perception — the agent is reading your data")

        if not st.session_state.get("async_mode"):
            # DEFAULT: synchronous — battle-tested; cards + voice stream progressively
            box = st.container()
            try:
                with st.spinner("Profiling…"):
                    readme_text = readme_up.read().decode("utf-8", "replace") if readme_up else ""
                    state = run_perception(
                        up, nl_request=nl.strip(), readme_text=readme_text,
                        filename=up.name,
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_runs"),
                        use_llm=use_llm)
                    _stream_ledger(state, box, already=0)
            except Exception as exc:  # noqa: BLE001
                st.session_state["phase"] = "idle"
                st.error(f"Perception failed: {exc}")
                st.stop()
            st.session_state["state"] = state
            st.session_state["phase"] = "confirm"
            st.session_state["journey_told"] = False
        else:
            # ASYNC: heavy work in a worker thread; the page never blocks, so
            # the websocket stays alive and the tab can never white-screen.
            if st.session_state.get("worker") is None:
                if "pending_csv" not in st.session_state:
                    st.session_state["pending_csv"] = up.getvalue()
                    st.session_state["pending_name"] = up.name
                    st.session_state["pending_readme"] = (
                        readme_up.read().decode("utf-8", "replace") if readme_up else "")
                    st.session_state["pending_nl"] = nl.strip()
                    st.session_state["pending_llm"] = use_llm
                payload = st.session_state

                def _percieve_job(ss):
                    import io as _io
                    try:
                        state = run_perception(
                            _io.BytesIO(ss["pending_csv"]),
                            nl_request=ss["pending_nl"],
                            readme_text=ss["pending_readme"],
                            filename=ss["pending_name"],
                            base_dir=str(Path(tempfile.gettempdir()) / "rra_runs"),
                            use_llm=ss["pending_llm"],
                            on_state=lambda stx: ss.__setitem__("state_live", stx))
                        ss["state"] = state
                    except Exception as exc:  # noqa: BLE001
                        ss["worker_error"] = f"Perception failed: {exc}"

                if st.session_state.get("pending_llm") and not st.session_state.get("warmed"):
                    st.info("🔥 Warming local models first (one-time on a cold "
                            "start) — the page stays live throughout.")
                    st.session_state["warmed"] = True

                    def _full_job(ss):
                        try:
                            from core.llm import warm_up
                            warm_up()
                        except Exception:  # noqa: BLE001
                            pass
                        _percieve_job(ss)
                    _start_worker(_full_job, payload)
                else:
                    _start_worker(_percieve_job, payload)

            @st.fragment(run_every="1.0s")
            def _perceive_poll():
                err = st.session_state.get("worker_error")
                if err:
                    st.session_state.update(phase="idle", worker=None)
                    for k in ("pending_csv", "pending_name", "pending_readme",
                              "pending_nl", "pending_llm"):
                        st.session_state.pop(k, None)
                    st.error(err)
                    st.rerun()
                _final = st.session_state.get("state")
                if _final is None and not _worker_alive() and \
                        st.session_state.get("state_live") is not None:
                    st.session_state["state"] = st.session_state["state_live"]
                    _final = st.session_state["state"]
                if _final is not None and not _worker_alive():
                    st.session_state["worker"] = None
                    st.session_state["phase"] = "confirm"
                    st.session_state["journey_told"] = False
                    st.rerun()
                stt = (st.session_state.get("state")
                       or st.session_state.get("state_live"))
                n = len(stt.ledger.entries) if stt is not None else 0
                last = stt.ledger.entries[-1].stage if (stt and n) else "starting"
                st.progress(min(0.1 + n * 0.11, 0.95),
                            text=f"🧠 {n} decisions logged — now in `{last}`…")
                if stt is not None:
                    for e in stt.ledger.entries[-4:]:
                        st.caption(f"`{e.stage}` — {e.decision}")
            _perceive_poll()
            st.stop()

    state = st.session_state.get("state")
    if _phase() != "idle" and state is None:
        st.session_state["phase"] = "idle"
        st.info("Session state was lost — please upload your file and run again.")
        st.stop()

    if _phase() == "confirm":
        from agents import stage_stories as SS
        if not st.session_state.get("journey_told"):
            st.markdown("## 🎬 The journey so far")
            _story_card("📥", "Ingest", SS.ingest_story(state),
                        script=SS.SPEAK["ingest"](state), stage_key="ingest")
            _story_card("🔎", "Profiling (EDA)", SS.eda_story(state),
                        script=SS.SPEAK["eda"](state), stage_key="eda")
            series = _daily_series(state)
            if series is not None and HAS_PLOTLY:
                cA, cB = st.columns([3, 2])
                with cA:
                    figj = go.Figure(go.Scatter(x=series.index, y=series.values,
                                                mode="lines",
                                                line=dict(color=TEAL, width=1.5)))
                    _show(_layout(figj, height=230, title="What I'm looking at — daily revenue"))
                with cB:
                    dow = series.groupby(series.index.dayofweek).mean()
                    figd = go.Figure(go.Bar(
                        x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(dow)],
                        y=dow.values, marker_color=AMBER))
                    _show(_layout(figd, height=230, title="The weekly pulse I detected"))
            _story_card("🧬", "Feature engineering", SS.features_story(state),
                        script=SS.SPEAK["features"](state), stage_key="features")
            fr = state.feature_report or {}
            imp = fr.get("importance")
            if imp and HAS_PLOTLY:
                rows_i = [{"feature": k,
                           "importance": (v.get("rf_importance", 0)
                                          if isinstance(v, dict) else float(v))}
                          for k, v in imp.items()]
                idf = pd.DataFrame(rows_i).sort_values("importance").tail(8)
                if len(idf):
                    figi = go.Figure(go.Bar(x=idf["importance"], y=idf["feature"],
                                            orientation="h", marker_color=TEAL))
                    _show(_layout(figi, height=60 + 32 * len(idf),
                                  title="Why each signal matters — predictive weight"))
            st.session_state["journey_told"] = True
        st.subheader("🧑‍⚖️ Confirm the plan (human-in-the-loop)")
        _story_card("🗺️", "The plan", SS.plan_story(state),
                    script=SS.SPEAK["plan"](state), stage_key="plan")
        dom = state.domain
        st.info(f"Detected domain: **{dom.get('profile', {}).get('display', dom.get('name'))}** "
                f"({dom.get('confidence', 0):.0%}) — evidence: "
                f"{', '.join(dom.get('evidence', [])[:5])}")

        cm = state.column_map
        cols = list(state.raw_df.columns)
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Date column", cols, key="sel_date",
                                index=cols.index(cm["date_column"]) if cm.get("date_column") in cols else 0)
        target_col = c2.selectbox("Revenue / target column", cols, key="sel_target",
                                  index=cols.index(cm["target_column"]) if cm.get("target_column") in cols else 0)
        id_col = c3.selectbox("Entity / customer column", ["(none)"] + cols, key="sel_id",
                              index=(cols.index(cm["id_column"]) + 1) if cm.get("id_column") in cols else 0)
        if cm.get("confidence", 1.0) < 0.7:
            st.warning(f"Column mapping confidence is low ({cm.get('confidence'):.0%}) — "
                       "please check the selections above.")

        intents = st.multiselect("Capabilities to run", options=list(INTENT_LABELS),
                                 key="intents",
                                 default=[i for i in state.intents if i in INTENT_LABELS],
                                 format_func=lambda i: INTENT_LABELS[i])

        if st.button("✅ Approve plan & run", type="primary", key="approve"):
            changed = (date_col != cm.get("date_column")
                       or target_col != cm.get("target_column")
                       or (None if id_col == "(none)" else id_col) != cm.get("id_column")
                       or set(intents) != set(state.intents))
            state.column_map.update(date_column=date_col, target_column=target_col,
                                    id_column=None if id_col == "(none)" else id_col)
            state.intents = [i for i in
                             ("segment", "anomaly", "leakage", "forecasting",
                              "rca", "whatif", "recommend") if i in intents]
            state.ledger.log(stage="plan_approval", agent="human",
                             decision="Plan approved" + (" with edits" if changed else " as proposed"),
                             reasoning=f"columns=({date_col},{target_col},{id_col}); "
                                       f"intents={state.intents}",
                             hitl_required=True, hitl_resolution="approved")
            st.session_state["phase"] = "execute"
        if _phase() == "confirm":
            st.stop()

    if _phase() == "execute":
        st.subheader("🤖 Reasoning — live ledger")
        box = st.container()
        n_before = len(state.ledger.entries)
        _stream_ledger(state, box, already=max(0, n_before - 1))
        from agents import stage_stories as SS
        cap_icons = {"segment": "👥", "anomaly": "🚨", "leakage": "💸",
                     "forecasting": "🏁", "rca": "🔍", "whatif": "🎛",
                     "recommend": "✅"}

        def _render_cap_card(intent, st_state, speak=True):
            fn = SS.CAPABILITY_STORIES.get(intent)
            if fn is None:
                return
            res = st_state.results.get(intent, {})
            if isinstance(res, dict) and "error" in res:
                _story_card(cap_icons.get(intent, "•"), intent,
                            f"This capability failed: `{res['error']}` — "
                            f"the rest of the run continues without it.",
                            speak=False)
                return
            if intent == "forecasting":
                _story_card("🏁", "The bake-off", SS.bakeoff_story(st_state),
                            speak=False)
            speak_fn = SS.SPEAK.get(intent)
            _story_card(cap_icons.get(intent, "•"),
                        intent.replace("_", " ").title(), fn(st_state),
                        speak=speak,
                        script=speak_fn(st_state) if speak and speak_fn else None,
                        stage_key=intent)

        if not st.session_state.get("async_mode"):
            try:
                with st.spinner("Planner → Executor → Critic at work…"):
                    run_capabilities(state, use_llm=use_llm,
                                     on_capability_done=lambda i, s2:
                                     _render_cap_card(i, s2))
                    summary = run_narration(state, use_llm=use_llm)
            except Exception as exc:  # noqa: BLE001
                st.session_state["phase"] = "confirm"
                st.error(f"Execution failed: {exc} — plan can be adjusted "
                         f"and re-approved.")
                st.stop()
            st.session_state["summary"] = summary
            st.session_state["phase"] = "done"
        else:
            # ASYNC PARALLEL: independent capabilities race in threads; cards
            # open the moment each lands; narration queues over them.
            if st.session_state.get("worker") is None and \
                    not st.session_state.get("caps_started"):
                st.session_state["caps_started"] = True
                st.session_state["cap_done"] = []
                st.session_state["spoken_caps"] = []
                payload = st.session_state

                def _exec_job(ss):
                    try:
                        from core.dispatcher import run_capabilities_parallel
                        run_capabilities_parallel(
                            ss["state"], use_llm=ss.get("pending_llm", False),
                            on_capability_done=lambda i, s2:
                            ss["cap_done"].append(i))
                        ss["summary"] = run_narration(
                            ss["state"], use_llm=ss.get("pending_llm", False))
                    except Exception as exc:  # noqa: BLE001
                        ss["worker_error"] = f"Execution failed: {exc}"
                _start_worker(_exec_job, payload)

            @st.fragment(run_every="1.0s")
            def _exec_poll():
                err = st.session_state.get("worker_error")
                if err:
                    st.session_state.update(phase="confirm", worker=None,
                                            caps_started=False)
                    st.error(err + " — plan can be adjusted and re-approved.")
                    st.rerun()
                stt = st.session_state["state"]
                done_caps = list(st.session_state.get("cap_done", []))
                total = max(len(stt.intents), 1)
                st.progress(min(len(done_caps) / total, 0.98),
                            text=f"⚡ {len(done_caps)}/{total} capabilities "
                                 f"complete — running in parallel…")
                for intent in done_caps:
                    speak = intent not in st.session_state["spoken_caps"]
                    if speak:
                        st.session_state["spoken_caps"].append(intent)
                    _render_cap_card(intent, stt, speak=speak)
                if not _worker_alive() and st.session_state.get("summary"):
                    st.session_state["worker"] = None
                    st.session_state["caps_started"] = False
                    st.session_state["phase"] = "done"
                    st.rerun()
            _exec_poll()
            st.stop()

    if _phase() == "done":
        from agents import stage_stories as SS
        r0 = state.results
        lk0 = r0.get("leakage", {})
        if lk0 and "error" not in lk0 and lk0.get("total_impact_estimate"):
            st.markdown(
                f"<div style='text-align:center;padding:28px 0 8px 0'>"
                f"<div style='font-size:20px;color:{SLATE}'>estimated recoverable revenue</div>"
                f"<div style='font-size:96px;font-weight:800;color:{AMBER};line-height:1.05'>"
                f"{lk0['total_impact_estimate']:,.0f}</div></div>",
                unsafe_allow_html=True)
            _speak(SS.SPEAK["hero"](state))
        _story_card("📜", "The audit trail", SS.narration_story(state), speak=False)
        st.success("✅ Analysis complete — open **📖 Findings** in the sidebar "
                   "for the story, **🔬 Evidence** for the charts, and "
                   "**🧠 Reasoning** for every decision the agent made.")
        r = state.results
        k = st.columns(3)
        lk = r.get("leakage", {})
        if lk and "error" not in lk:
            k[0].metric("💸 Money at risk", f"{lk.get('total_impact_estimate', 0):,.0f}")
        fc = r.get("forecasting", {})
        if fc and "error" not in fc:
            k[1].metric("📈 Forecast", fc.get("winner_label", "—"),
                        fc.get("verdict"), delta_color="off")
        an = r.get("anomaly", {})
        if an and "error" not in an:
            k[2].metric("🚨 High-confidence anomalies",
                        an.get("counts", {}).get("high", 0))


# ============================================================ FINDINGS SCREEN
def screen_findings():
    if _need_results():
        return
    state = st.session_state["state"]
    r = state.results
    st.title("📖 Findings")

    k = st.columns(3)
    lk = r.get("leakage") if "error" not in r.get("leakage", {}) else None
    fc = r.get("forecasting") if "error" not in r.get("forecasting", {}) else None
    an = r.get("anomaly") if "error" not in r.get("anomaly", {}) else None
    if lk:
        k[0].metric("💸 Money at risk", f"{lk.get('total_impact_estimate', 0):,.0f}",
                    f"{lk.get('candidate_count', 0)} records", delta_color="inverse")
    if fc:
        ens = fc.get("ensemble")
        sub = (f"blend MAPE {ens['blend_mape']:.2f}%" if ens else fc.get("verdict"))
        k[1].metric("📈 Forecast", fc.get("winner_label", "—"), sub, delta_color="off")
    if an:
        c = an.get("counts", {})
        k[2].metric("🚨 Data health", f"{c.get('high', 0)} high-risk",
                    f"{c.get('medium', 0)} medium", delta_color="off")

    if fc and fc.get("verdict") == "escalate":
        st.warning(f"⚠️ **The Critic declined to certify a forecast.** "
                   f"{fc.get('critic_reasoning', '')} — best attempt "
                   f"({fc.get('best_available', '—')}) is shown in Evidence "
                   f"with that caveat. Refusing to overstate confidence is the "
                   f"safeguard working.")
    elif fc and fc.get("ensemble"):
        e = fc["ensemble"]
        st.success(f"🤝 The Critic chose an **ensemble**: {' + '.join(e['members'])} "
                   f"(weights {e['weights']}) — blended MAPE **{e['blend_mape']:.2f}%** "
                   f"beats any single model; error correlation {e['error_corr']} < 0.9.")

    st.markdown("---")
    st.markdown(st.session_state.get("summary", "_No narrative available._"))
    st.markdown("---")
    st.caption("The numbers behind every claim are in 🔬 Evidence; the decision "
               "trail behind every number is in 🧠 Reasoning.")


# ====================================================== DATA & FEATURES SCREEN
def screen_data_features():
    if _need_results():
        return
    state = st.session_state["state"]
    df = _frame(state)
    cm = state.column_map
    st.title("📊 Data & Features")

    # ---- KPI band ----
    tgt, dcol, idc = cm.get("target_column"), cm.get("date_column"), cm.get("id_column")
    ing = state.ingest_report
    k = st.columns(5)
    if tgt in df.columns:
        k[0].metric("Total billed", f"{df[tgt].sum():,.0f}")
    k[1].metric("Records", f"{ing.get('rows', len(df)):,}")
    if idc and idc in df.columns:
        k[2].metric("Customers", f"{df[idc].nunique():,}")
    if dcol and dcol in df.columns:
        dd = pd.to_datetime(df[dcol], errors="coerce")
        k[3].metric("Date range", f"{(dd.max() - dd.min()).days} days")
    miss = df.isna().mean().mean()
    k[4].metric("Missing / dupes", f"{miss:.1%} / {df.duplicated().mean():.1%}")

    # ---- EDA charts ----
    c1, c2 = st.columns([3, 2])
    series = _daily_series(state)
    with c1:
        if series is not None and HAS_PLOTLY:
            fig = go.Figure(go.Scatter(x=series.index, y=series.values,
                                       mode="lines", line=dict(color=TEAL, width=1.6),
                                       hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>"))
            _show(_layout(fig, height=300, title=f"{tgt} over time (daily)"))
    with c2:
        seg_col = next((c for c in df.columns
                        if c.lower() in ("segment", "category", "class", "type")
                        and df[c].nunique() <= 12), None)
        if seg_col and tgt in df.columns and HAS_PLOTLY:
            agg = df.groupby(seg_col)[tgt].sum().sort_values()
            fig = go.Figure(go.Bar(x=agg.values, y=agg.index.astype(str),
                                   orientation="h", marker_color=AMBER,
                                   text=[f"{v:,.0f}" for v in agg.values],
                                   textposition="outside", textfont_color=TEXT,
                                   hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>"))
            _show(_layout(fig, height=300, title=f"{tgt} by {seg_col}"))

    c3, c4 = st.columns(2)
    with c3:
        if idc and idc in df.columns and tgt in df.columns and HAS_PLOTLY:
            top = df.groupby(idc)[tgt].sum().nlargest(10).sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index.astype(str),
                                   orientation="h", marker_color=SLATE,
                                   hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>"))
            _show(_layout(fig, height=330, title="Top 10 customers by billed"))
    with c4:
        mv = df.isna().mean().sort_values(ascending=False)
        mv = mv[mv > 0].head(8)
        if len(mv) and HAS_PLOTLY:
            fig = go.Figure(go.Bar(x=mv.values * 100, y=mv.index, orientation="h",
                                   marker_color=RED,
                                   text=[f"{v:.2%}" for v in mv.values],
                                   textposition="outside", textfont_color=TEXT,
                                   hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>"))
            _show(_layout(fig, height=330, title="Missing values by column (%)"))
        else:
            st.success("No missing values detected.")

    # ---- EDA facts strip ----
    ts = state.eda_report.get("timeseries") or {}
    seas = ts.get("seasonality") or {}
    facts = []
    if seas:
        facts.append(f"seasonality: period {seas.get('period', '?')}, "
                     f"strength {seas.get('strength', 0):.2f}")
    if ts.get("trend"):
        facts.append(f"trend: {ts['trend']}")
    if ts.get("stationarity"):
        facts.append(f"stationarity: {ts['stationarity'].get('conclusion', '?')} "
                     f"(ADF p={ts['stationarity'].get('adf_pvalue', 0):.3f})")
    if facts:
        st.caption("EDA: " + " · ".join(facts))

    # ---- Feature engineering report ----
    st.markdown("### 🧬 Feature engineering")
    fr = state.feature_report or {}
    rules = fr.get("applied_rules", {})
    n_feats = sum(len(v) for v in rules.values())
    c5, c6, c7 = st.columns(3)
    c5.metric("Rule groups applied", len(rules))
    c6.metric("Features created", n_feats)
    fa = state.results.get("feature_analysis", state.feature_report or {})
    keep = fr.get("top_features") or fa.get("top_features") or []
    c7.metric("Features that matter", len(keep) if keep else "—")
    for grp, feats in rules.items():
        st.markdown(f"- **{grp}** → `{'`, `'.join(feats[:8])}`"
                    + (f" (+{len(feats) - 8} more)" if len(feats) > 8 else ""))
    imp = fr.get("importance") or fa.get("importance")
    if imp and HAS_PLOTLY:
        if isinstance(imp, dict):
            rows_i = [{"feature": k,
                       "importance": (v.get("rf_importance", 0) if isinstance(v, dict)
                                      else float(v))}
                      for k, v in imp.items()]
        else:
            rows_i = list(imp)
        idf = pd.DataFrame(rows_i)
        if {"feature", "importance"} <= set(idf.columns) and len(idf):
            idf = idf.sort_values("importance", ascending=True).tail(10)
            fig = go.Figure(go.Bar(x=idf["importance"], y=idf["feature"],
                                   orientation="h", marker_color=TEAL,
                                   hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>"))
            _show(_layout(fig, height=90 + 34 * len(idf),
                          title="Feature importance (top 10)"))


# ============================================================ EVIDENCE SCREEN
def screen_evidence():
    if _need_results():
        return
    state = st.session_state["state"]
    r = state.results
    st.title("🔬 Evidence")

    # ---- Forecast ----
    from agents import stage_stories as SS
    fc = r.get("forecasting")
    if fc and "error" not in fc:
        with st.expander("📈 Forecast — bake-off & forward view", expanded=True):
            st.markdown("> " + SS.verdict_story(state))
            if fc.get("verdict") == "escalate":
                st.warning(f"⚠️ Escalated: {fc.get('critic_reasoning', '')} — the "
                           f"chart below is the **best attempt "
                           f"({fc.get('best_available', '—')}), not a certified "
                           f"forecast**.")
            series = _daily_series(state)
            if series is not None and fc.get("forecast") and HAS_PLOTLY:
                hist = series.tail(120)
                fdates = pd.date_range(series.index.max(),
                                       periods=len(fc["forecast"]) + 1,
                                       freq=pd.infer_freq(series.index) or "D")[1:]
                fig = go.Figure()
                fig.add_scatter(x=hist.index, y=hist.values, name="Actual",
                                mode="lines", line=dict(color=SLATE, width=2))
                dash = "dot" if fc.get("verdict") == "escalate" else "dash"
                fig.add_scatter(x=[hist.index[-1], *fdates],
                                y=[hist.values[-1], *fc["forecast"]],
                                name=f"Forecast · {fc.get('winner_label', '')}",
                                mode="lines+markers",
                                line=dict(color=TEAL if fc.get("verdict") != "escalate"
                                          else AMBER, width=3, dash=dash),
                                marker=dict(size=5))
                fig.add_vrect(x0=hist.index[-1], x1=fdates[-1], fillcolor=TEAL,
                              opacity=0.06, line_width=0)
                _show(_layout(fig, hovermode="x unified"))
            if fc.get("ensemble"):
                e = fc["ensemble"]
                st.success(f"🤝 Ensemble: {' + '.join(e['members'])} · weights "
                           f"{e['weights']} · blended MAPE {e['blend_mape']:.2f}% "
                           f"· error corr {e['error_corr']}")
            ok = [m for m in fc.get("metrics", [])
                  if not m.get("error") and m.get("mape") is not None]
            bad = [m for m in fc.get("metrics", [])
                   if m.get("error") or m.get("mape") is None]
            if ok and HAS_PLOTLY:
                mdf = pd.DataFrame(ok).sort_values("mape")
                members = (fc["ensemble"]["members"] if fc.get("ensemble")
                           else [fc.get("winner"), fc.get("best_available")])
                colors = [TEAL if m in members else GRID for m in mdf["model"]]
                fig = go.Figure(go.Bar(
                    x=mdf["mape"], y=mdf["model"] + "  ·  " + mdf["family"],
                    orientation="h", marker_color=colors,
                    text=mdf["mape"].map(lambda v: f"{v:.2f}%"),
                    textposition="outside", textfont_color=TEXT,
                    hovertemplate="%{y}<br>MAPE %{x:.2f}%<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                _show(_layout(fig, height=70 + 42 * len(mdf),
                              title="Bake-off — backtest MAPE (lower is better)"))
            if bad:
                st.caption("Did not produce valid metrics: "
                           + ", ".join(f"{m['model']} ({m.get('error') or 'no valid backtest'})"
                                       for m in bad))

            # ---- 🏆 Winner audit card ----
            wc = fc.get("winner_card")
            if wc:
                with st.container(border=True):
                    st.markdown(f"#### 🏆 Winner audit — "
                                f"{fc.get('winner_label', wc.get('name'))}")
                    st.markdown(f"**Why selected:** {wc.get('why_selected', '—')}")
                    cc = st.columns(4)
                    cc[0].metric("MAPE", f"{wc['mape']:.2f}%"
                                 if wc.get("mape") is not None else "—")
                    cc[1].metric("vs naive floor",
                                 f"+{wc['margin_vs_baseline']:.2f}pp"
                                 if wc.get("margin_vs_baseline") is not None else "—")
                    cc[2].metric("vs runner-up",
                                 f"+{wc['margin_vs_runner_up']:.2f}pp"
                                 if wc.get("margin_vs_runner_up") is not None else "—")
                    cc[3].metric("Attempts", wc.get("attempts", 1))
                    prep = ", ".join(f"{k}={v}" for k, v in
                                     (wc.get("preprocessing") or {}).items()) or "none"
                    st.caption(f"Preprocessing: {prep}")
                    for cv in wc.get("caveats", []):
                        st.warning(f"⚠️ {cv}")

            # ---- 🤝 Ensemble panel — ALWAYS shown, fired or not ----
            ee = fc.get("ensemble_eval")
            if ee and not fc.get("ensemble"):
                st.info(f"🤝 **Ensemble considered, not chosen:** {ee.get('reason', '—')}")

            # ---- 📋 Full model roster — every registry model, its status, its why ----
            roster = fc.get("roster")
            if roster:
                STATUS = {"ran": "✅ ran", "selected": "✅ ran",
                          "data_gated": "🚫 data does not support",
                          "not_selected": "⏸ eligible, not selected",
                          "failed": "❌ failed"}
                rows_r = []
                for name, v in roster.items():
                    rows_r.append({
                        "model": name, "family": v.get("family", "?"),
                        "status": STATUS.get(v.get("status"), v.get("status")),
                        "MAPE %": (f"{v['mape']:.2f}" if v.get("mape") is not None else "—"),
                        "why": (v.get("reason") or "")[:140],
                    })
                order = {"✅ ran": 0, "⏸ eligible, not selected": 1,
                         "🚫 data does not support": 2, "❌ failed": 3}
                rows_r.sort(key=lambda r: (order.get(r["status"], 9), r["model"]))
                st.markdown("**Model accountability board** — all "
                            f"{len(rows_r)} registry models:")
                st.dataframe(pd.DataFrame(rows_r), width='stretch', hide_index=True)

    # ---- Anomalies ----
    an = r.get("anomaly")
    if an and "error" not in an:
        c = an.get("counts", {})
        with st.expander(f"🚨 Anomalies — {c.get('high', 0)} high / "
                         f"{c.get('medium', 0)} medium", expanded=True):
            st.markdown("> " + SS.anomaly_story(state))
            heavy = (c.get("high", 0) + c.get("medium", 0)) > 150
            tiers = st.multiselect("Show tiers", ["high", "medium", "review"],
                                   default=["high"] if heavy else ["high", "medium"],
                                   key="tiers")
            if heavy:
                st.caption("Large volume — defaulting to high tier; add tiers above.")
            ros = an.get("roster")
            if ros:
                bits = []
                for d, v in ros.items():
                    if v.get("status") == "ran":
                        bits.append(f"✅ {d} ({v.get('votes_cast', 0)} votes)")
                    else:
                        bits.append(f"🚫 {d} — {v.get('reason', 'skipped')}")
                st.caption("Detector board: " + " · ".join(bits))
            rows = [f for f in an.get("flagged", []) if f.get("tier") in tiers]
            if rows and HAS_PLOTLY:
                adf = pd.DataFrame(rows)
                adf["date"] = pd.to_datetime(adf["date"], errors="coerce")
                series = _daily_series(state)
                fig = go.Figure()
                if series is not None:
                    fig.add_scatter(x=series.index, y=series.values, mode="lines",
                                    name="Daily revenue",
                                    line=dict(color=GRID, width=1.5), hoverinfo="skip")
                for tier in ("review", "medium", "high"):
                    sub = adf[adf["tier"] == tier]
                    if sub.empty:
                        continue
                    fig.add_scatter(
                        x=sub["date"], y=sub["billed_amount"], mode="markers",
                        name=f"{tier} ({len(sub)})",
                        marker=dict(color=TIER_COLORS[tier], size=6 + sub["votes"] * 2.2,
                                    opacity=0.55 if tier == "review" else 0.9,
                                    line=dict(width=0)),
                        customdata=sub[["entity", "votes"]].astype(str).values,
                        text=[("; ".join(str(x) for x in a[:2]) if isinstance(a, list) else str(a))
                              for a in sub["attribution"]],
                        hovertemplate=("<b>%{customdata[0]}</b> · %{x|%Y-%m-%d}"
                                       "<br>amount %{y:,.0f} · votes %{customdata[1]}"
                                       "<br>%{text}<extra></extra>"))
                _show(_layout(fig, height=430,
                              title="Anomaly constellation — dot size = detector votes"))
            if rows:
                adf2 = pd.DataFrame(rows)
                adf2["attribution"] = adf2["attribution"].apply(
                    lambda a: "; ".join(str(x) for x in a[:2]) if isinstance(a, list) else str(a))
                adf2["voted_by"] = adf2["voted_by"].apply(
                    lambda v: ", ".join(v) if isinstance(v, list) else str(v))
                st.dataframe(adf2.head(300), width='stretch', hide_index=True)

    # ---- Leakage ----
    lk = r.get("leakage")
    if lk and "error" not in lk:
        with st.expander(f"💸 Leakage — {lk.get('total_impact_estimate', 0):,.0f} "
                         f"recoverable", expanded=True):
            st.markdown("> " + SS.leakage_story(state))
            all_rules = lk.get("rules_fired", {})
            clean = [k2 for k2, v in all_rules.items() if not v.get("hits")]
            if clean:
                st.caption("Rules checked and CLEAN (0 hits): " + ", ".join(clean)
                           + " — a zero is evidence too.")
            fired = {k2: v for k2, v in all_rules.items() if v.get("hits")}
            if fired and HAS_PLOTLY:
                fdf = (pd.DataFrame([{"rule": k2, "impact": v["impact_total"],
                                      "hits": v["hits"], "desc": v.get("description", "")}
                                     for k2, v in fired.items()]).sort_values("impact"))
                fig = go.Figure(go.Bar(
                    x=fdf["impact"], y=fdf["rule"], orientation="h",
                    marker=dict(color=fdf["impact"],
                                colorscale=[[0, "#7A5A22"], [1, AMBER]], showscale=False),
                    text=fdf.apply(lambda x: f"{x['impact']:,.0f} · {x['hits']} hits", axis=1),
                    textposition="outside", textfont_color=TEXT,
                    customdata=fdf[["desc"]].values,
                    hovertemplate="<b>%{y}</b><br>impact %{x:,.0f}<br>%{customdata[0]}<extra></extra>"))
                _show(_layout(fig, height=90 + 52 * len(fdf), title="Impact by leakage rule"))
            st.dataframe(pd.DataFrame(lk.get("candidates", [])).head(300),
                         width='stretch', hide_index=True)

    # ---- Root cause ----
    rc = r.get("rca")
    if rc and "error" not in rc:
        with st.expander("🔍 Root cause", expanded=False):
            st.markdown("> " + SS.rca_story(state))
            fw = rc.get("evidence", {}).get("focus_window", {})
            st.markdown(f"**Focus window:** {fw.get('start')} → {fw.get('end')} "
                        f"(Δ {fw.get('delta_vs_previous', 0):,.0f} vs previous window)")
            drivers = rc.get("evidence", {}).get("decomposition_top_drivers", {})
            dims = [d for d in drivers if drivers.get(d)]
            if dims and HAS_PLOTLY:
                cols_ = st.columns(min(2, len(dims)))
                for i, dim in enumerate(dims[:2]):
                    ddf = pd.DataFrame(drivers[dim]).sort_values("delta")
                    fig = go.Figure(go.Bar(
                        x=ddf["delta"], y=ddf["value"].astype(str), orientation="h",
                        marker_color=[RED if v < 0 else TEAL for v in ddf["delta"]],
                        text=ddf["delta"].map(lambda v: f"{v:+,.0f}"),
                        textposition="outside", textfont_color=TEXT,
                        hovertemplate="<b>%{y}</b><br>Δ %{x:,.0f}<extra></extra>"))
                    with cols_[i % 2]:
                        _show(_layout(fig, height=90 + 46 * len(ddf), title=f"Δ by {dim}"))
            for h in rc.get("hypotheses", []):
                st.markdown(f"- **{h.get('confidence', 0):.0%}** — {h.get('hypothesis')}"
                            + (f"  \n  <sub>{', '.join(str(x) for x in h.get('evidence', [])[:4])}</sub>"
                               if h.get("evidence") else ""), unsafe_allow_html=True)

    # ---- Segments ----
    sg = r.get("segment")
    if sg and "error" not in sg:
        with st.expander(f"👥 Segments — k={sg.get('k')}", expanded=False):
            st.markdown("> " + SS.segment_story(state))
            profs = pd.DataFrame(sg.get("profiles", []))
            if not profs.empty and HAS_PLOTLY:
                palette = [TEAL, AMBER, RED, SLATE, "#7C6BD9", "#4E9AD6"]
                fig = go.Figure()
                for i, row in profs.iterrows():
                    fig.add_scatter(
                        x=[row["rfm_recency_days"]], y=[row["rfm_monetary"]],
                        mode="markers+text", name=str(row["persona"]),
                        text=[f"S{row['segment_id']} · {row['size']}"],
                        textposition="top center", textfont_color=TEXT,
                        marker=dict(size=18 + row["size"] * 1.4,
                                    color=palette[i % len(palette)], opacity=0.75,
                                    line=dict(color=TEXT, width=1)),
                        hovertemplate=(f"<b>{row['persona']}</b><br>{row['size']} customers"
                                       "<br>recency %{x:.0f} d · monetary %{y:,.0f}"
                                       "<extra></extra>"))
                fig.update_xaxes(title="Recency (days since last bill)")
                fig.update_yaxes(title="Monetary (total billed)")
                _show(_layout(fig, height=430, title="Segments — bubble size = customers"))
            if not profs.empty:
                st.dataframe(profs, width='stretch', hide_index=True)

    # ---- What-if ----
    wf = r.get("whatif")
    if wf and "error" not in wf:
        with st.expander("🎛 What-if scenarios", expanded=False):
            st.markdown(f"Baseline (next {wf.get('horizon_days', '?')} days): "
                        f"**{wf.get('baseline_total', 0):,.0f}** — "
                        f"<sub>{wf.get('assumption', '')}</sub>", unsafe_allow_html=True)
            sdf = pd.DataFrame(wf.get("scenarios", []))
            if not sdf.empty and HAS_PLOTLY:
                sdf = sdf.sort_values("delta_vs_baseline")
                fig = go.Figure(go.Bar(
                    x=sdf["delta_vs_baseline"], y=sdf["scenario"], orientation="h",
                    marker_color=[RED if v < 0 else TEAL for v in sdf["delta_vs_baseline"]],
                    text=sdf.apply(lambda x: f"{x['delta_vs_baseline']:+,.0f} "
                                             f"({x['delta_pct']:+.1f}%)", axis=1),
                    textposition="outside", textfont_color=TEXT,
                    hovertemplate="<b>%{y}</b><br>Δ %{x:,.0f}<extra></extra>"))
                _show(_layout(fig, height=90 + 52 * len(sdf),
                              title="Scenario deltas vs baseline"))
            if not sdf.empty:
                st.dataframe(sdf, width='stretch', hide_index=True)

    # ---- Actions ----
    rec = r.get("recommend")
    if rec and "error" not in rec:
        with st.expander("✅ Recommended actions", expanded=True):
            for x in rec.get("recommendations", []):
                with st.container(border=True):
                    st.markdown(f"**{x.get('action')}**")
                    cc = st.columns(3)
                    cc[0].caption(f"effort: {x.get('effort', '?')}")
                    cc[1].caption(f"confidence: {x.get('confidence', 0):.0%}")
                    cc[2].caption(f"traces to: `{x.get('traces_to', '?')}`")


# =========================================================== REASONING SCREEN
def screen_reasoning():
    if _need_results():
        return
    state = st.session_state["state"]
    st.title("🧠 Reasoning ledger")
    st.caption("Append-only. Every decision, its reasoning, its evidence, and "
               "which engine made it. ⚙️ deterministic · 🤖 LLM · 🧑 human.")
    agents = sorted({e.agent for e in state.ledger.entries})
    pick = st.multiselect("Filter by engine", agents, default=agents, key="agent_filter")
    n_llm = sum(1 for e in state.ledger.entries
                if e.agent not in ("deterministic", "human"))
    n_h = sum(1 for e in state.ledger.entries if e.agent == "human")
    st.markdown(f"**{len(state.ledger.entries)} decisions** — "
                f"🤖 {n_llm} LLM · ⚙️ {len(state.ledger.entries) - n_llm - n_h} "
                f"deterministic · 🧑 {n_h} human")
    for e in state.ledger.entries:
        if e.agent in pick:
            st.markdown(_entry_md(e), unsafe_allow_html=True)
            st.divider()


# ============================================================== EXPORT SCREEN
def screen_export():
    if _need_results():
        return
    state = st.session_state["state"]
    summary = st.session_state.get("summary", "")
    st.title("📄 Export")
    cache_key = f"exports_{state.run_id}"
    if cache_key not in st.session_state:
        with st.spinner("Rendering deck, report and PDF…"):
            import tempfile as _tf
            from agents.exporter import build_pptx, build_docx, build_pdf
            base = Path(_tf.gettempdir())
            files = {}
            for name, fn, ext in (("deck", build_pptx, "pptx"),
                                  ("report", build_docx, "docx"),
                                  ("chartbook", build_pdf, "pdf")):
                try:
                    fp = str(base / f"rra_{name}_{state.run_id}.{ext}")
                    fn(state, summary, fp)
                    files[ext] = Path(fp).read_bytes()
                except Exception as exc:  # noqa: BLE001
                    files[ext] = None
                    st.warning(f"{ext.upper()} export failed: {exc}")
            st.session_state[cache_key] = files
    files = st.session_state[cache_key]
    c1, c2, c3 = st.columns(3)
    if files.get("pptx"):
        c1.download_button("⬇ Executive deck (.pptx)", files["pptx"],
                           file_name=f"rra_findings_{state.run_id}.pptx", key="dl_pptx",
                           width='stretch')
    if files.get("docx"):
        c2.download_button("⬇ Full report (.docx)", files["docx"],
                           file_name=f"rra_report_{state.run_id}.docx", key="dl_docx",
                           width='stretch')
    if files.get("pdf"):
        c3.download_button("⬇ Chart book (.pdf)", files["pdf"],
                           file_name=f"rra_chartbook_{state.run_id}.pdf", key="dl_pdf",
                           width='stretch')
    with st.expander("Raw / machine-readable"):
        report_md = build_report_markdown(state, summary)
        st.download_button("Markdown report", report_md,
                           file_name=f"rra_report_{state.run_id}.md", key="dl_md")
        st.download_button("Decision ledger (JSONL)",
                           Path(state.ledger.path).read_text(encoding="utf-8"),
                           file_name="ledger.jsonl", key="dl_jsonl")


# ================================================================== SIDEBAR
with st.sidebar:
    st.title("🧭 Revenue Reasoning Agent")
    nav = st.radio("Screens", SCREENS, key="nav", label_visibility="collapsed")
    st.markdown("---")

    up = readme_up = None
    nl = ""
    use_llm = False
    if nav == "▶ Run":
        up = st.file_uploader("Revenue / billing CSV", type=["csv"], key="csv")
        readme_up = st.file_uploader("Context doc (optional)", type=["md", "txt"],
                                     key="readme")
        nl = st.text_area("What do you want to know?", key="nl",
                          value="Find revenue leakage, forecast next month, explain "
                                "why revenue moved, segment my customers and "
                                "recommend actions.",
                          height=90)
        if "ollama_up" not in st.session_state:
            st.session_state["ollama_up"] = ollama_available()
        ollama_up = st.session_state["ollama_up"]
        oc1, oc2 = st.columns([3, 1])
        oc1.markdown(f"**Ollama:** {'🟢 reachable' if ollama_up else '🔴 not detected'}")
        if oc2.button("↻", key="ollama_refresh", help="Re-check Ollama"):
            st.session_state["ollama_up"] = ollama_available()
            st.rerun()
        use_llm = st.toggle("Use LLM agents", value=ollama_up, key="use_llm",
                            help="Off = fully deterministic fallbacks (still complete).")
        speed = st.radio("LLM speed", ["⚡ Fast (llama3.2:3b)",
                                       "🎯 Quality (llama3.1:8b)"],
                         index=0, key="llm_speed", horizontal=True,
                         help="Fast is 2-3× quicker on CPU and plenty for the "
                              "structured perception/planning calls.")
        from core.llm import set_reasoner
        set_reasoner("llama3.2:3b" if speed.startswith("⚡") else "llama3.1:8b")
        st.toggle("🔊 Narrate the run", value=False, key="voice_on",
                  help="Browser voice gives a running commentary (demo mode).")
        st.toggle("⚡ Experimental async engine", value=False, key="async_mode",
                  help="Parallel capabilities + live progress bar. Default off: "
                       "the synchronous engine is the battle-tested path.")
        if st.session_state.get("voice_on"):
            st.selectbox("Voice", ["Female", "Male",
                                   "Duet (Narrator + Critic)", "Auto"],
                         index=0, key="voice_pref",
                         format_func=lambda v: v)
            if st.session_state.get("voice_pref", "").startswith("Duet"):
                st.session_state["voice_pref"] = "Duet"
        if use_llm and not ollama_up:
            st.warning("Ollama not reachable — calls will fall back per-agent.")
        run_btn = st.button("▶ Run analysis", type="primary", key="run",
                            disabled=up is None, width='stretch')
        if run_btn and up is not None:
            st.session_state.pop("state", None)
            st.session_state["phase"] = "perceive"
        if _phase() in ("confirm", "done"):
            if st.button("↺ New run", key="newrun", width='stretch'):
                for k in ("phase", "state", "summary"):
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        use_llm = bool(st.session_state.get("use_llm", False))
        if _phase() == "done":
            state = st.session_state.get("state")
            if state is not None:
                lk = state.results.get("leakage", {})
                if lk and "error" not in lk:
                    st.metric("💸 Money at risk",
                              f"{lk.get('total_impact_estimate', 0):,.0f}")
                st.caption(f"Run {state.run_id} · "
                           f"{state.domain.get('name', '?')} domain")

# ================================================================== ROUTER
if nav == "▶ Run":
    screen_run(up, readme_up, nl, use_llm)
elif nav == "📖 Findings":
    screen_findings()
elif nav == "📊 Data & Features":
    screen_data_features()
elif nav == "🔬 Evidence":
    screen_evidence()
elif nav == "🧠 Reasoning":
    screen_reasoning()
else:
    screen_export()
