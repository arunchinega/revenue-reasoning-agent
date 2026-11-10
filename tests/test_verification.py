"""Dependency-light tests for deterministic verification."""
from types import SimpleNamespace
from core.verification import verify_results


def test_rejects_nonfinite_forecast():
    state = SimpleNamespace(results={"forecasting": {
        "forecast": [1.0, float("nan")], "metrics": [], "verdict": "accept_single"
    }}, ledger=None)
    report = verify_results(state)
    assert report["status"] == "rejected"
    assert report["rejected_count"] == 1


def test_qualifies_unproven_recoverability():
    state = SimpleNamespace(results={"leakage": {
        "total_impact_estimate": 100.0
    }}, ledger=None)
    report = verify_results(state)
    assert report["status"] == "qualified"
    assert report["verified_count"] == 1
    assert report["unverified_count"] == 1


def test_no_results():
    state = SimpleNamespace(results={}, ledger=None)
    assert verify_results(state)["status"] == "not_applicable"
