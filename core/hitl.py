"""Explicit human approval gate shared by UI and dispatcher.

This module does not grant approval implicitly. Legacy programmatic callers
remain supported unless they opt into require_approval=True.
"""
from __future__ import annotations

from core.state import RunState


class ApprovalRequired(RuntimeError):
    pass


def resolve_plan(state: RunState, decision: str, *,
                 intents: list[str] | None = None,
                 column_map: dict | None = None) -> None:
    """Record an explicit approval or rejection; reject never mutates plan."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be approved or rejected")
    if decision == "approved":
        if intents is not None:
            state.intents = list(intents)
        if column_map is not None:
            state.column_map.update(column_map)
    state.plan["human_approval"] = decision
    state.ledger.log(
        stage="plan_approval", agent="human",
        decision=f"Plan {decision}",
        reasoning="Explicit human decision before capability execution",
        hitl_required=True, hitl_resolution=decision,
    )


def require_approved_plan(state: RunState) -> None:
    """Fail closed: missing, rejected, or malformed approval blocks execution."""
    if state.plan.get("human_approval") != "approved":
        raise ApprovalRequired("Human approval required before executing capabilities")
