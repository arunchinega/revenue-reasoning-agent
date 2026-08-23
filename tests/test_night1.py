"""End-to-end smoke test: Stage 0 → 1 → 1.5 → 1.6 on synthetic utilities data."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.state import RunState
from stages.ingest import run_ingest
from stages.eda_profiler import run_eda
from stages.feature_engine import run_feature_engineering
from stages.feature_analysis import run_feature_analysis
from tests.make_demo_data import make_utilities_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv_path)

    state = RunState(nl_request="find revenue leakage and forecast next quarter")
    state.init_run(base_dir=str(Path(tempfile.gettempdir()) / "rra_runs"))

    run_ingest(state, csv_path, filename="demo_utilities.csv")
    run_eda(state)
    run_feature_engineering(state)
    run_feature_analysis(state)

    # --- assertions -----------------------------------------------------------
    assert state.ingest_report["gates_passed"]
    assert state.ingest_report["date_columns"] == ["bill_date"]
    assert "billed_amount" in state.ingest_report["numeric_columns"]

    eda = state.eda_report
    assert eda["target_guess"] == "billed_amount", eda["target_guess"]
    ts = eda["timeseries"]
    assert ts and ts["frequency"] == "daily", ts
    assert ts["seasonality"].get("detected"), f"weekly seasonality not detected: {ts['seasonality']}"
    assert eda["categorical"]["segment"]["segmentation_viable"]

    fr = state.feature_report
    assert fr["n_features_added"] >= 8, fr["n_features_added"]
    assert fr["detected_roles"]["id_col"] == "customer_id"
    assert fr["detected_roles"]["price_col"] == "tariff_rate"
    assert "billed_vs_expected_delta" in state.feature_df.columns  # leakage signal
    assert fr["top_features"], "importance ranking empty"
    # expected_revenue ≈ target → must be caught by leakage guard, not ranked top
    assert "expected_revenue" in fr["leakage_suspects_dropped"], fr["leakage_suspects_dropped"]

    assert len(state.ledger.entries) == 4  # one per stage
    print("=== LEDGER ===")
    for e in state.ledger.entries:
        print(f"[{e.stage}] {e.decision}\n    reasoning: {e.reasoning}\n    evidence: {e.evidence}")

    print("\n=== EDA timeseries ===")
    print(json.dumps(ts, indent=2, default=str))
    print("\n=== top features ===", fr["top_features"][:8])
    print("\nALL NIGHT-1 CHECKS PASSED ✅  run dir:", state.run_dir)


if __name__ == "__main__":
    main()
