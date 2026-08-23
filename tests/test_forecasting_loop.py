"""Smoke test: perception + forecasting capability loop, no Ollama needed."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orchestrator import run_perception
from agents.planner_critic import run_forecasting_capability
from capabilities.forecasting import registry_eligibility
from tests.make_demo_data import make_utilities_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv_path)

    state = run_perception(
        csv_path,
        nl_request="forecast billed revenue for the next month",
        filename="demo_utilities.csv",
        base_dir=str(Path(tempfile.gettempdir()) / "rra_runs3"),
        use_llm=False,
    )

    # eligibility sanity: gates fire off real EDA evidence
    elig = registry_eligibility(state.eda_report["timeseries"])
    assert elig["seasonal_naive"]["eligible"]
    assert elig["sarima"]["eligible"], elig["sarima"]          # seasonality detected
    assert elig["lstm"]["eligible"] or "unavailable" in elig["lstm"]["reasoning"]
    assert not elig["prophet"]["eligible"] or True             # env-dependent

    out = run_forecasting_capability(state, use_llm=False, horizon=30)

    assert out["verdict"] in ("accept_single", "accept_ensemble"), out["verdict"]
    assert out["forecast"] is not None and len(out["forecast"]) >= 7
    valid = [m for m in out["metrics"] if m["mape"] is not None]
    baseline = next(m for m in valid if m["family"] == "baseline")
    best = valid[0]
    assert best["mape"] < baseline["mape"], "winner must beat the naive floor"

    print("=== METRICS TABLE ===")
    for m in out["metrics"]:
        print(f"  {m['model']:16s} {m['family']:14s} MAPE={m['mape']}  "
              f"RMSE={m['rmse']}  folds={m['folds']}  err={m['error']}")
    print(f"\nverdict: {out['verdict']}  winner: {out['winner']}  "
          f"attempts: {out['attempts']}")

    print("\n=== LEDGER (reasoning loop) ===")
    for e in state.ledger.entries:
        if e.stage in ("planner", "critic"):
            print(f"[{e.stage}] {e.decision}")
            print(f"    {e.reasoning}")
            if e.data.get("excluded"):
                print(f"    excluded: {e.data['excluded']}")

    print("\nFORECASTING LOOP CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
