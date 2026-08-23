"""
core/state.py — Shared agent state + Reasoning Ledger.

The ledger is append-only JSONL: every decision by any agent (or human)
is recorded with reasoning, evidence citations, and confidence.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ----------------------------------------------------------------------------
# Reasoning Ledger
# ----------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    stage: str                          # e.g. "eda", "planner", "critic"
    agent: str                          # "deterministic" | "llama-3.1-8b" | "human" | ...
    decision: str                       # short statement of what was decided
    reasoning: str = ""                 # why
    evidence: list[str] = field(default_factory=list)   # e.g. ["eda.seasonality.weekly"]
    confidence: Optional[float] = None  # 0..1 where meaningful
    hitl_required: bool = False
    hitl_resolution: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)  # any structured payload
    ts: str = ""
    entry_id: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if not self.entry_id:
            self.entry_id = uuid.uuid4().hex[:8]


class ReasoningLedger:
    """Append-only decision log. Writes JSONL to disk, keeps in-memory list for UI."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "ledger.jsonl"
        self.entries: list[LedgerEntry] = []

    def log(self, **kwargs) -> LedgerEntry:
        entry = LedgerEntry(**kwargs)
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry

    def by_stage(self, stage: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.stage == stage]

    def to_markdown(self) -> str:
        """Export ledger as a readable audit trail (for report export)."""
        lines = ["# Reasoning Ledger", ""]
        for e in self.entries:
            conf = f" · confidence {e.confidence:.2f}" if e.confidence is not None else ""
            lines.append(f"## [{e.ts}] {e.stage} — {e.agent}{conf}")
            lines.append(f"**Decision:** {e.decision}")
            if e.reasoning:
                lines.append(f"**Reasoning:** {e.reasoning}")
            if e.evidence:
                lines.append(f"**Evidence:** {', '.join(e.evidence)}")
            if e.hitl_required:
                lines.append(f"**Human-in-the-loop:** {e.hitl_resolution or 'pending'}")
            lines.append("")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Run State — the single object passed between stages / agents
# ----------------------------------------------------------------------------

@dataclass
class RunState:
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))
    run_dir: Optional[Path] = None

    # user inputs
    nl_request: str = ""
    readme_text: str = ""               # optional context doc — never required

    # dataframes (kept out of serialization; referenced by attribute only)
    raw_df: Any = None
    clean_df: Any = None
    feature_df: Any = None

    # structured evidence store — everything agents cite
    ingest_report: dict = field(default_factory=dict)
    eda_report: dict = field(default_factory=dict)
    feature_report: dict = field(default_factory=dict)

    # decisions
    domain: dict = field(default_factory=dict)      # {name, confidence, evidence}
    intents: list[str] = field(default_factory=list)
    column_map: dict = field(default_factory=dict)  # {date_column, target_column, id_column, feature_columns, confidence}
    plan: dict = field(default_factory=dict)        # per-intent plans from Planner
    results: dict = field(default_factory=dict)     # per-capability results

    # ledger (constructed on init_run)
    ledger: Optional[ReasoningLedger] = None

    def init_run(self, base_dir: str | Path = "runs") -> "RunState":
        self.run_dir = Path(base_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = ReasoningLedger(self.run_dir)
        return self

    def save_report(self, name: str, payload: dict) -> Path:
        """Persist a structured report (eda_report.json etc.) into the run dir."""
        path = self.run_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return path
