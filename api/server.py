"""api/server.py — thin FastAPI wrapper over the RRA engine. No Streamlit.
Run:  python -m uvicorn api.server:app --host 127.0.0.1 --port 8600
UI:   http://127.0.0.1:8600/
"""
from __future__ import annotations

import io
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core.quiet  # noqa: F401,E402
from core.orchestrator import run_perception  # noqa: E402
from core.dispatcher import run_capabilities  # noqa: E402
from agents.narrator import run_narration  # noqa: E402
from agents import stage_stories as SS  # noqa: E402

app = FastAPI(title="RRA API")
RUN: dict = {"phase": "idle", "state": None, "summary": "", "error": None,
             "use_llm": False, "wav": None, "clips": [], "clip_q": []}


def _clip_worker():
    """Synthesise queued (text, speaker) items into small WAV clips, in order.
    One subprocess per clip via temp .py — logged, never silent."""
    import json as _json
    import subprocess
    import sys as _sys
    while True:
        if not RUN["clip_q"]:
            import time as _t
            _t.sleep(0.4)
            if RUN["phase"] == "idle" and not RUN["clip_q"]:
                continue
            continue
        text, speaker = RUN["clip_q"].pop(0)
        try:
            tmp = Path(tempfile.mkdtemp(prefix="rra_clip_"))
            (tmp / "i.json").write_text(_json.dumps([[text, speaker]]),
                                        encoding="utf-8")
            root = Path(__file__).resolve().parents[1]
            script = tmp / "s.py"
            lines = [
                "import json, sys",
                "from pathlib import Path",
                f"sys.path.insert(0, {str(root)!r})",
                "from agents.voicebox import synth_narration",
                f"it = json.loads(Path({str(tmp / 'i.json')!r})"
                ".read_text(encoding='utf-8'))",
                "w = synth_narration([tuple(x) for x in it], duet=True)",
                f"Path({str(tmp / 'o.wav')!r}).write_bytes(w or b'')",
            ]
            script.write_text("\n".join(lines), encoding="utf-8")
            subprocess.run([_sys.executable, str(script)], timeout=90,
                           capture_output=True)
            wav = tmp / "o.wav"
            if wav.exists() and wav.stat().st_size > 1000:
                RUN["clips"].append(wav.read_bytes())
                print(f"[voice] clip {len(RUN['clips'])} ready "
                      f"({wav.stat().st_size:,}b): {text[:50]}…", flush=True)
            else:
                print(f"[voice] clip FAILED: {text[:50]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[voice] clip EXC {type(e).__name__}: {e}", flush=True)


threading.Thread(target=_clip_worker, daemon=True).start()


def _say(text, speaker="narrator"):
    RUN["clip_q"].append((text, speaker))


def _ledger(state):
    return [{"stage": e.stage, "agent": e.agent, "decision": e.decision,
             "reasoning": (e.reasoning or "")[:220]}
            for e in state.ledger.entries] if state else []


@app.post("/api/run")
async def start_run(file: UploadFile, nl: str = Form(""), use_llm: bool = Form(False)):
    data = await file.read()
    RUN.update(phase="perceiving", state=None, summary="", error=None,
               use_llm=use_llm, wav=None, clips=[], clip_q=[])

    def job():
        try:
            st = run_perception(io.BytesIO(data), nl_request=nl,
                                filename=file.filename,
                                base_dir=str(Path(tempfile.gettempdir()) / "rra_api"),
                                use_llm=use_llm,
                                on_state=lambda s: RUN.__setitem__("state", s))
            RUN.update(state=st, phase="confirm")
            _say(SS.SPEAK["ingest"](st))
            _say(SS.SPEAK["eda"](st))
            _say(SS.SPEAK["features"](st))
            _say(SS.SPEAK["plan"](st), "critic")
        except Exception as e:  # noqa: BLE001
            RUN.update(error=f"{type(e).__name__}: {e}", phase="idle")
    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}


@app.post("/api/approve")
def approve():
    st = RUN["state"]
    if st is None:
        return JSONResponse({"ok": False}, status_code=400)
    RUN["phase"] = "executing"
    st.ledger.log(stage="plan_approval", agent="human",
                  decision="Plan approved as proposed",
                  reasoning="approved via web UI", hitl_required=True,
                  hitl_resolution="approved")

    def _voice_job(st):
        try:
            import pythoncom  # Windows COM init for SAPI in a thread
            pythoncom.CoInitialize()
        except Exception:  # noqa: BLE001
            pass
        try:
            from agents.voicebox import synth_narration
            order = ["ingest", "eda", "features", "forecasting", "anomaly",
                     "leakage", "rca", "segment", "whatif", "recommend", "hero"]
            items = [(SS.SPEAK[k](st), SS.DUET_SPEAKER.get(k)) for k in order
                     if k in SS.SPEAK and (k in ("ingest", "eda", "features",
                                                 "hero") or k in st.results)]
            RUN["wav"] = synth_narration(items, duet=True)
        except Exception:  # noqa: BLE001
            pass

    def job():
        try:
            def _on_cap(intent, s2):
                fn = SS.SPEAK.get(intent)
                if fn:
                    _say(fn(s2), SS.DUET_SPEAKER.get(intent, "narrator"))
            run_capabilities(st, use_llm=RUN["use_llm"],
                             on_capability_done=_on_cap)
            RUN["summary"] = run_narration(st, use_llm=RUN["use_llm"])
            RUN["phase"] = "done"          # results FIRST — never wait on audio
            threading.Thread(target=_voice_job, args=(st,), daemon=True).start()
        except Exception as e:  # noqa: BLE001
            RUN.update(error=f"{type(e).__name__}: {e}", phase="confirm")
    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    st = RUN["state"]
    out = {"phase": RUN["phase"], "error": RUN["error"],
           "ledger": _ledger(st), "clips": len(RUN["clips"])}
    if st is not None:
        out["domain"] = st.domain.get("name")
        out["confidence"] = st.domain.get("confidence", 0)
        out["intents"] = st.intents
        out["columns"] = st.column_map
        if RUN["phase"] == "confirm":
            out["plan_story"] = SS.plan_story(st)
        if RUN["phase"] == "done":
            r = st.results
            lk = r.get("leakage", {})
            fc = r.get("forecasting", {})
            out["hero"] = lk.get("total_impact_estimate")
            # ---- chart payloads ----
            try:
                import pandas as pd
                cm = st.column_map
                df = st.feature_df if st.feature_df is not None else st.raw_df
                d = df[[cm["date_column"], cm["target_column"]]].copy()
                d[cm["date_column"]] = pd.to_datetime(d[cm["date_column"]])
                ser = d.groupby(cm["date_column"])[cm["target_column"]] \
                    .sum().sort_index().tail(90)
                out["series"] = {"dates": [str(x.date()) for x in ser.index],
                                 "values": [round(float(v), 2) for v in ser.values]}
                out["forecast"] = fc.get("forecast", [])[:30]
                out["rules"] = [{"rule": k, "impact": round(v.get("impact_total", 0)),
                                 "hits": v.get("hits", 0)}
                                for k, v in lk.get("rules_fired", {}).items()
                                if v.get("hits")]
                an = r.get("anomaly", {})
                pts = [f for f in an.get("flagged", [])
                       if f.get("tier") in ("high", "medium")][:400]
                out["anoms"] = [{"d": str(f.get("date"))[:10],
                                 "v": float(f.get(cm["target_column"], 0)),
                                 "t": f.get("tier")} for f in pts]
                out["bakeoff"] = [{"m": m["model"], "mape": m.get("mape")}
                                  for m in fc.get("metrics", [])
                                  if m.get("mape") is not None]
                out["winner_name"] = fc.get("winner")
            except Exception as e:  # noqa: BLE001
                print(f"[charts] payload error: {e}", flush=True)
            out["stories"] = {k: fn(st) for k, fn in SS.CAPABILITY_STORIES.items()
                              if k in r and "error" not in r.get(k, {})}
            out["winner"] = fc.get("winner_label")
            out["metrics"] = fc.get("metrics", [])[:6]
            out["summary"] = RUN["summary"]
            out["has_audio"] = RUN["wav"] is not None
    return out


@app.get("/api/clip/{i}")
def clip(i: int):
    if i < 0 or i >= len(RUN["clips"]):
        return Response(status_code=404)
    return Response(RUN["clips"][i], media_type="audio/wav")


@app.get("/chart.js")
def chartjs():
    return Response((Path(__file__).parent.parent / "web" / "chart.js")
                    .read_bytes(), media_type="application/javascript")


@app.get("/api/audio")
def audio():
    if RUN["wav"] is None:
        return Response(status_code=404)
    return Response(RUN["wav"], media_type="audio/wav")


@app.get("/")
def index():
    return HTMLResponse((Path(__file__).parent.parent / "web" / "index.html")
                        .read_text(encoding="utf-8"))
