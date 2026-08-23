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
             "use_llm": False, "wav": None}


def _ledger(state):
    return [{"stage": e.stage, "agent": e.agent, "decision": e.decision,
             "reasoning": (e.reasoning or "")[:220]}
            for e in state.ledger.entries] if state else []


@app.post("/api/run")
async def start_run(file: UploadFile, nl: str = Form(""), use_llm: bool = Form(False)):
    data = await file.read()
    RUN.update(phase="perceiving", state=None, summary="", error=None,
               use_llm=use_llm, wav=None)

    def job():
        try:
            st = run_perception(io.BytesIO(data), nl_request=nl,
                                filename=file.filename,
                                base_dir=str(Path(tempfile.gettempdir()) / "rra_api"),
                                use_llm=use_llm,
                                on_state=lambda s: RUN.__setitem__("state", s))
            RUN.update(state=st, phase="confirm")
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

    def job():
        try:
            run_capabilities(st, use_llm=RUN["use_llm"])
            RUN["summary"] = run_narration(st, use_llm=RUN["use_llm"])
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
            RUN["phase"] = "done"
        except Exception as e:  # noqa: BLE001
            RUN.update(error=f"{type(e).__name__}: {e}", phase="confirm")
    threading.Thread(target=job, daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    st = RUN["state"]
    out = {"phase": RUN["phase"], "error": RUN["error"], "ledger": _ledger(st)}
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
            out["stories"] = {k: fn(st) for k, fn in SS.CAPABILITY_STORIES.items()
                              if k in r and "error" not in r.get(k, {})}
            out["winner"] = fc.get("winner_label")
            out["metrics"] = fc.get("metrics", [])[:6]
            out["summary"] = RUN["summary"]
            out["has_audio"] = RUN["wav"] is not None
    return out


@app.get("/api/audio")
def audio():
    if RUN["wav"] is None:
        return Response(status_code=404)
    return Response(RUN["wav"], media_type="audio/wav")


@app.get("/")
def index():
    return HTMLResponse((Path(__file__).parent.parent / "web" / "index.html")
                        .read_text(encoding="utf-8"))
