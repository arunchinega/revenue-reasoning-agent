"""HITL safety regression tests for the actual dispatcher gate."""
import tempfile
import unittest
from unittest.mock import patch

from core.state import RunState
from core.hitl import ApprovalRequired, resolve_plan
from core.dispatcher import run_capabilities, run_capabilities_parallel


class HumanApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = RunState(intents=["forecasting"], column_map={"target_column": "revenue"})
        self.state.init_run(self.tmp.name)

    def test_missing_approval_blocks_execution(self):
        with patch("core.dispatcher._RUNNERS", {"forecasting": lambda s, llm: self.fail("executed")}):
            with self.assertRaises(ApprovalRequired):
                run_capabilities(self.state, use_llm=False, require_approval=True)

    def test_rejection_blocks_execution_and_preserves_plan(self):
        resolve_plan(self.state, "rejected", intents=["anomaly"],
                     column_map={"target_column": "other"})
        self.assertEqual(self.state.intents, ["forecasting"])
        self.assertEqual(self.state.column_map["target_column"], "revenue")
        with self.assertRaises(ApprovalRequired):
            run_capabilities(self.state, use_llm=False, require_approval=True)

    def test_approval_with_edits_executes_only_selected_intent(self):
        resolve_plan(self.state, "approved", intents=["anomaly"],
                     column_map={"target_column": "net_revenue"})
        calls = []
        with patch("core.dispatcher._RUNNERS", {"anomaly": lambda s, llm: calls.append("anomaly")}):
            run_capabilities(self.state, use_llm=False, require_approval=True)
        self.assertEqual(calls, ["anomaly"])
        self.assertEqual(self.state.column_map["target_column"], "net_revenue")
        self.assertEqual(self.state.plan["human_approval"], "approved")

    def test_parallel_dispatch_also_blocks_rejection(self):
        resolve_plan(self.state, "rejected")
        with self.assertRaises(ApprovalRequired):
            run_capabilities_parallel(self.state, use_llm=False, require_approval=True)


if __name__ == "__main__":
    unittest.main()
