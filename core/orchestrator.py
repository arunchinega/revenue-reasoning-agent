"""
core/orchestrator.py — sequences stages 0 → 2.5, then dispatches capabilities.

Kept as a plain, debuggable sequencer for the perception+understanding phase;
the LangGraph Planner/Executor/Critic loop attaches at run_capabilities()
(day-block work). `use_llm=False` runs the whole front half deterministically —
useful for tests and for machines without Ollama.
"""
from __future__ import annotations

import io
from pathlib import Path

from core.state import RunState
from stages.ingest import run_ingest
from stages.eda_profiler import run_eda
from stages.feature_engine import run_feature_engineering
from stages.feature_analysis import run_feature_analysis
from stages.domain_detect import run_domain_detection
from stages.intent_detect import run_intent_detection
from stages.column_map import run_column_mapping


def ollama_available(timeout: float = 1.5) -> bool:
    """Fast, never-blocking availability probe (UI calls this every rerun)."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags",
                                    timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def run_perception(source: str | Path | io.BytesIO,
                   nl_request: str = "",
                   readme_text: str = "",
                   filename: str = "upload.csv",
                   base_dir: str = "runs",
                   use_llm: bool | None = None,
                   on_state=None) -> RunState:
    """Stages 0 → 2.5. Returns state ready for the Planner.
    on_state(state) fires immediately after construction so a UI can watch
    the ledger grow live."""
    if use_llm is None:
        use_llm = ollama_available()

    state = RunState(nl_request=nl_request, readme_text=readme_text)
    state.init_run(base_dir=base_dir)
    if on_state is not None:
        try:
            on_state(state)
        except Exception:  # noqa: BLE001
            pass
    state.ledger.log(
        stage="orchestrator", agent="deterministic",
        decision=f"Run started (LLM {'enabled' if use_llm else 'DISABLED — deterministic fallbacks only'})",
        reasoning="Ollama reachable" if use_llm else "Ollama not reachable or explicitly disabled",
    )

    run_ingest(state, source, filename=filename)
    run_eda(state)
    run_feature_engineering(state)
    run_feature_analysis(state)
    run_domain_detection(state, use_llm=use_llm)
    run_intent_detection(state, use_llm=use_llm)
    run_column_mapping(state, use_llm=use_llm)
    return state


def pending_hitl(state: RunState) -> list:
    """Entries awaiting human confirmation — UI polls this."""
    return [e for e in state.ledger.entries
            if e.hitl_required and e.hitl_resolution is None]
