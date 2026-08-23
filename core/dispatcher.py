"""
core/dispatcher.py — routes detected intents to capability modules in
dependency order. This is where the capability plug-ins meet the chassis.
"""
from __future__ import annotations

from core.state import RunState
from agents.planner_critic import run_forecasting_capability
from capabilities.anomaly import run_anomaly_capability
from capabilities.leakage import run_leakage_capability
from capabilities.rca import run_rca_capability
from capabilities.segmentation import (
    run_recommend_capability, run_segmentation_capability, run_whatif_capability,
)

# canonical dependency-safe execution order (intent_detect emits this order)
_RUNNERS = {
    "segment": lambda s, llm: run_segmentation_capability(s, use_llm=llm),
    "anomaly": lambda s, llm: run_anomaly_capability(s),
    "leakage": lambda s, llm: run_leakage_capability(s),
    "forecasting": lambda s, llm: run_forecasting_capability(s, use_llm=llm),
    "rca": lambda s, llm: run_rca_capability(s, use_llm=llm),
    "whatif": lambda s, llm: run_whatif_capability(s),
    "recommend": lambda s, llm: run_recommend_capability(s, use_llm=llm),
}


def run_capabilities(state: RunState, use_llm: bool = True,
                     on_capability_done=None) -> RunState:
    """on_capability_done(intent, state) fires after each capability finishes
    (success or failure) — the UI uses it to render stage cards live."""
    for intent in state.intents:
        runner = _RUNNERS.get(intent)
        if runner is None:
            continue
        try:
            runner(state, use_llm)
        except Exception as e:  # noqa: BLE001 — one capability failing never kills the run
            state.results[intent] = {"error": f"{type(e).__name__}: {e}"}
            state.ledger.log(
                stage=intent, agent="deterministic",
                decision=f"capability FAILED: {type(e).__name__}",
                reasoning=str(e)[:300],
            )
        if on_capability_done is not None:
            try:
                on_capability_done(intent, state)
            except Exception:  # noqa: BLE001 — UI callback must never kill the run
                pass
    return state


INDEPENDENT = ("segment", "anomaly", "leakage", "forecasting")
DEPENDENT = ("rca", "whatif", "recommend")   # need earlier results


def run_capabilities_parallel(state: RunState, use_llm: bool = True,
                              on_capability_done=None,
                              max_workers: int = 3) -> RunState:
    """Independent capabilities race concurrently; dependents run after.
    Same failure isolation and callback contract as run_capabilities."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one(intent: str) -> str:
        runner = _RUNNERS.get(intent)
        if runner is None:
            return intent
        try:
            runner(state, use_llm)
        except Exception as e:  # noqa: BLE001
            state.results[intent] = {"error": f"{type(e).__name__}: {e}"}
            state.ledger.log(stage=intent, agent="deterministic",
                             decision=f"capability FAILED: {type(e).__name__}",
                             reasoning=str(e)[:300])
        return intent

    indep = [i for i in state.intents if i in INDEPENDENT]
    dep = [i for i in state.intents if i in DEPENDENT]
    if indep:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_one, i): i for i in indep}
            for fut in as_completed(futs):
                done = fut.result()
                if on_capability_done is not None:
                    try:
                        on_capability_done(done, state)
                    except Exception:  # noqa: BLE001
                        pass
    for intent in dep:
        _one(intent)
        if on_capability_done is not None:
            try:
                on_capability_done(intent, state)
            except Exception:  # noqa: BLE001
                pass
    return state
