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

    def test_no_results(self):
        state = SimpleNamespace(results={}, ledger=None)
        self.assertEqual(verify_results(state)["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
