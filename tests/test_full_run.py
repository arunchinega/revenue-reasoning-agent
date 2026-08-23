"""Grand smoke test: all 7 capabilities end-to-end, deterministic mode."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from tests.make_demo_data import make_utilities_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv_path)

    state = run_perception(
        csv_path,
        nl_request="find revenue leakage, forecast next month, explain why revenue "
                   "moved, segment my customers, run what-if scenarios and recommend actions",
        filename="demo_utilities.csv",
        base_dir=str(Path(tempfile.gettempdir()) / "rra_full"),
        use_llm=False,
    )
    assert set(state.intents) == {"segment", "anomaly", "leakage", "forecasting",
                                  "rca", "whatif", "recommend"}, state.intents
    run_capabilities(state, use_llm=False)

    for cap in state.intents:
        res = state.results.get(cap)
        assert res is not None, f"{cap} produced nothing"
        assert "error" not in res, f"{cap} errored: {res.get('error')}"

    assert state.results["forecasting"]["verdict"].startswith("accept")
    assert state.results["leakage"]["total_impact_estimate"] > 0
    assert state.results["segment"]["k"] >= 2
    assert state.results["whatif"]["scenarios"]
    assert state.results["recommend"]["recommendations"]
    assert state.results["rca"]["hypotheses"]

    print("=== FULL-RUN LEDGER ===")
    for e in state.ledger.entries:
        conf = f" ({e.confidence})" if e.confidence is not None else ""
        print(f"[{e.stage}]{conf} {e.decision}")

    recs = state.results["recommend"]["recommendations"]
    print("\n=== RECOMMENDATIONS ===")
    for r in recs:
        print(f"  - {r['action']}  [effort={r['effort']} conf={r['confidence']}]")
        print(f"      traces_to: {r['traces_to']}")

    print(f"\nledger entries: {len(state.ledger.entries)} | run dir: {state.run_dir}")
    print("ALL 7 CAPABILITIES PASSED ✅")


if __name__ == "__main__":
    main()
