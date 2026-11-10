"""Guard against accidental removal of the Streamlit rejection control.

This is a source-level UI contract test, not an automated browser test.
"""
import ast
from pathlib import Path
import unittest


class StreamlitApprovalControlTests(unittest.TestCase):
    def test_rejection_button_and_execution_gate_present(self):
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        button_labels = [
            node.args[0].value for node in calls
            if isinstance(node.func, ast.Attribute)
            and node.func.attr == "button"
            and node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertIn("❌ Reject plan", button_labels)
        self.assertIn("✅ Approve plan & run", button_labels)
        self.assertIn('resolve_plan(state, "rejected")', source)
        self.assertIn('resolve_plan(state, "approved")', source)
        self.assertIn("require_approval=True", source)


if __name__ == "__main__":
    unittest.main()
