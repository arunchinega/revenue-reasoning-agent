"""Dependency-light unittest cases for deterministic verification."""
import unittest
from types import SimpleNamespace

from core.verification import verify_results


class VerificationTests(unittest.TestCase):
    def test_rejects_nonfinite_forecast(self):
        state = SimpleNamespace(results={"forecasting": {
            "forecast": [1.0, float("nan")], "metrics": [], "verdict": "accept_single"
        }}, ledger=None)
        report = verify_results(state)
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["rejected_count"], 1)

    def test_qualifies_unproven_recoverability(self):
        state = SimpleNamespace(results={"leakage": {
            "total_impact_estimate": 100.0
        }}, ledger=None)
        report = verify_results(state)
        self.assertEqual(report["status"], "qualified")
        self.assertEqual(report["verified_count"], 1)
        self.assertEqual(report["unverified_count"], 1)

    def test_rule_overlap_requires_reconciliation(self):
        state = SimpleNamespace(results={"leakage": {
            "total_impact_estimate": 90.0,
            "rules_fired": {
                "a": {"applicable": True, "impact_total": 80.0},
                "b": {"applicable": True, "impact_total": 30.0},
            },
        }}, ledger=None)
        report = verify_results(state)
        check = next(c for c in report["checks"]
                     if c["claim"] == "leakage_reconciliation")
        self.assertEqual(check["status"], "unverified")

    def test_impossible_net_exceeding_gross_rejected(self):
        state = SimpleNamespace(results={"leakage": {
            "total_impact_estimate": 120.0,
            "rules_fired": {
                "a": {"applicable": True, "impact_total": 80.0},
                "b": {"applicable": True, "impact_total": 30.0},
            },
        }}, ledger=None)
        self.assertEqual(verify_results(state)["status"], "rejected")

    def test_forecast_without_feature_manifest_is_qualified(self):
        state = SimpleNamespace(results={"forecasting": {
            "forecast": [1.0, 2.0], "metrics": [], "verdict": "escalate"
        }}, ledger=None)
        report = verify_results(state)
        self.assertEqual(report["status"], "qualified")
        self.assertTrue(any(c["claim"] == "forecast_feature_availability"
                            for c in report["checks"]))

    def test_no_results(self):
        state = SimpleNamespace(results={}, ledger=None)
        self.assertEqual(verify_results(state)["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
