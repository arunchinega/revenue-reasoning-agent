"""Regression tests for record-level financial reconciliation."""
import unittest
import pandas as pd
from core.state import RunState, ReasoningLedger
from capabilities.leakage import run_leakage_capability
from tempfile import TemporaryDirectory


class LeakageReconciliationTests(unittest.TestCase):
    def test_overlapping_rules_deduplicate_per_record(self):
        # Two identical invoices: both underbilled by 20, second also duplicate (80).
        df = pd.DataFrame({
            "customer_id": ["A", "A"],
            "bill_date": ["2026-01-01", "2026-01-01"],
            "billed_amount": [80.0, 80.0],
            "expected_revenue": [100.0, 100.0],
            "billed_vs_expected_delta": [-20.0, -20.0],
        })
        with TemporaryDirectory() as directory:
            state = RunState(raw_df=df)
            state.ledger = ReasoningLedger(directory)
            state.column_map = {
                "id_column": "customer_id",
                "date_column": "bill_date",
                "target_column": "billed_amount",
            }
            state.domain = {"name": "utilities", "profile": {
                "leakage_rules": ["billed_vs_expected_delta", "duplicate_billing"]
            }}
            result = run_leakage_capability(state)
            reconciliation = result["reconciliation"]
            self.assertEqual(result["candidate_count"], 2)
            self.assertAlmostEqual(reconciliation["gross_rule_impact"], 120.0)
            self.assertAlmostEqual(reconciliation["deduplicated_impact"], 100.0)
            self.assertAlmostEqual(reconciliation["overlap_impact"], 20.0)
            self.assertEqual(reconciliation["overlap_record_count"], 1)
            self.assertTrue(reconciliation["reconciled"])


if __name__ == "__main__":
    unittest.main()
