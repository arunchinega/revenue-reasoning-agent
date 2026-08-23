"""Smoke test: anomaly consensus + leakage rules must find the planted defects."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orchestrator import run_perception
from capabilities.anomaly import run_anomaly_capability
from capabilities.leakage import run_leakage_capability
from tests.make_demo_data import make_utilities_csv


def main() -> None:
    csv_path = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
    make_utilities_csv(csv_path)

    state = run_perception(
        csv_path,
        nl_request="find revenue leakage and billing anomalies",
        filename="demo_utilities.csv",
        base_dir=str(Path(tempfile.gettempdir()) / "rra_runs_al"),
        use_llm=False,
    )
    assert set(state.intents) >= {"leakage", "anomaly"}

    an = run_anomaly_capability(state)
    assert an["counts"]["high"] + an["counts"]["medium"] >= 3, an["counts"]
    assert len(an["detectors_run"]) >= 4, an["detectors_run"]
    # planted 8x consumption spikes must surface with kwh attribution
    spike_hits = [r for r in an["flagged"]
                  if any("kwh" in a for a in r["attribution"])]
    assert spike_hits, "consumption spikes not attributed to kwh"

    lk = run_leakage_capability(state)
    fired = {k for k, v in lk["rules_fired"].items() if v.get("applicable") and v.get("hits")}
    # planted: under-billing (0.4x), zero-billed-with-usage, tariff misapplication
    assert "billed_vs_expected_delta" not in (), None
    assert any(r in fired for r in ("billed_vs_expected_delta",)), fired
    assert "zero_billed_with_usage" in fired or "unbilled_usage" in fired, fired
    assert "tariff_misapplication" in fired, fired
    assert lk["candidate_count"] >= 8, lk["candidate_count"]
    assert lk["total_impact_estimate"] > 0
    high_conf = [c for c in lk["candidates"] if c["confidence"] == "high"]
    assert high_conf, "no high-confidence leakage candidates"

    print("=== ANOMALY ===")
    print("detectors:", an["detectors_run"], "| errors:", an["detector_errors"])
    print("counts:", an["counts"])
    for r in an["flagged"][:5]:
        print(f"  row {r['row_index']} [{r['tier']}] votes={r['votes']} "
              f"by={r['voted_by']} attr={r['attribution']}")

    print("\n=== LEAKAGE ===")
    for k, v in lk["rules_fired"].items():
        if v.get("applicable"):
            print(f"  rule {k}: hits={v['hits']} impact={v['impact_total']:,.0f}")
    print(f"total impact estimate: {lk['total_impact_estimate']:,.0f} "
          f"across {lk['candidate_count']} candidates")
    for c in lk["candidates"][:5]:
        print(f"  {c.get('entity')} {c.get('date', '')[:10]} [{c['confidence']}] "
              f"rules={c['rules']} votes={c['statistical_votes']} "
              f"impact={c['impact_estimate']:,.0f}")

    print("\n=== LEDGER (money story) ===")
    for e in state.ledger.entries:
        if e.stage in ("anomaly", "leakage"):
            print(f"[{e.stage}] {e.decision}\n    {e.reasoning}")

    print("\nANOMALY + LEAKAGE CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
