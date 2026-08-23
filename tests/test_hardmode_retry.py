"""Hard-mode test: outlier-polluted data MUST force the Critic's retry path.

Asserts the exact demo sequence:
  attempt 1 → retry (with outlier diagnosis + winsorize change)
  attempt 2 → accept (winner beats the naive floor after cleanup)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orchestrator import run_perception
from agents.planner_critic import run_forecasting_capability
from tests.make_hardmode_data import make_hardmode_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "hardmode_utilities.csv")
    make_hardmode_csv(csv_path)

    state = run_perception(
        csv_path,
        nl_request="forecast revenue for next month",
        filename="hardmode_utilities.csv",
        base_dir=str(Path(tempfile.gettempdir()) / "rra_hard"),
        use_llm=False,
    )
    # EDA must see the outlier pollution — that's what arms the Critic's diagnosis
    assert state.eda_report["numeric"]["billed_amount"]["outlier_pct_iqr"] > 3

    out = run_forecasting_capability(state, use_llm=False)

    critic_entries = [e for e in state.ledger.entries if e.stage == "critic"]
    assert len(critic_entries) >= 2, "retry loop never executed"
    assert "retry" in critic_entries[0].decision, critic_entries[0].decision
    assert "winsoriz" in critic_entries[0].reasoning.lower(), critic_entries[0].reasoning
    assert out["verdict"] in ("accept_single", "accept_ensemble"), out["verdict"]
    assert out["attempts"] == 2, out["attempts"]

    # winner must genuinely beat the baseline after cleanup
    valid = [m for m in out["metrics"] if m["mape"] is not None]
    baseline = next(m for m in valid if m["family"] == "baseline")
    assert valid[0]["mape"] < baseline["mape"]

    print("=== HARD-MODE REASONING SEQUENCE ===")
    for e in state.ledger.entries:
        if e.stage in ("planner", "critic"):
            print(f"[{e.stage}] {e.decision}")
            print(f"    {e.reasoning}")
    print(f"\nfinal: {out['verdict']} → {out['winner']} in {out['attempts']} attempts")
    print("HARD-MODE RETRY CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
