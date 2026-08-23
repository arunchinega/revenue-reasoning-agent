"""Smoke test: perception phase 0 → 2.5 deterministic (use_llm=False)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orchestrator import run_perception, pending_hitl
from tests.make_demo_data import make_utilities_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv_path)

    state = run_perception(
        csv_path,
        nl_request="find unbilled consumption and tariff mismatch issues, "
                   "then forecast demand for next quarter and tell me why revenue dipped in June",
        filename="demo_utilities.csv",
        base_dir=str(Path(tempfile.gettempdir()) / "rra_runs2"),
        use_llm=False,
    )

    # domain: utilities via schema signals (kwh, tariff, bill_date, meter absent but enough)
    assert state.domain["name"] == "utilities", state.domain
    assert "unbilled_usage" in state.domain["profile"]["leakage_rules"]

    # intents: leakage (domain vocab "unbilled consumption", "tariff mismatch"),
    # forecasting ("forecast"), rca ("why") → anomaly auto-added as rca dependency
    assert set(state.intents) >= {"leakage", "forecasting", "rca", "anomaly"}, state.intents
    # execution order: anomaly before rca
    assert state.intents.index("anomaly") < state.intents.index("rca")

    cm = state.column_map
    assert cm["target_column"] == "billed_amount"
    assert cm["date_column"] == "bill_date"
    assert cm["id_column"] == "customer_id"
    assert cm["confidence"] >= 0.8

    print("=== LEDGER (understanding phase) ===")
    for e in state.ledger.entries:
        flag = " [HITL]" if e.hitl_required else ""
        conf = f" ({e.confidence:.2f})" if e.confidence is not None else ""
        print(f"[{e.stage}]{conf}{flag} {e.decision}")
    print("\npending HITL:", [(e.stage, e.decision) for e in pending_hitl(state)])
    print("\nPERCEPTION PHASE CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
