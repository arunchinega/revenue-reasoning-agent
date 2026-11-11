"""Source-level regression guard for stable Streamlit execution rendering.

This is not a browser test; manually reproduce the prior failure in Chrome.
"""
import ast
from pathlib import Path
import unittest


class StableRenderingTests(unittest.TestCase):
    def test_no_nested_streamlit_callbacks_during_capability_execution(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
        screen = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "screen_run")
        calls = [n for n in ast.walk(screen) if isinstance(n, ast.Call)]
        dispatch = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "run_capabilities"]
        self.assertEqual(len(dispatch), 1)
        keywords = {kw.arg: kw.value for kw in dispatch[0].keywords}
        self.assertNotIn("on_capability_done", keywords)
        self.assertIsInstance(keywords["require_approval"], ast.Constant)
        self.assertTrue(keywords["require_approval"].value)

    def test_ledger_renderer_does_not_replace_log_method(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text())
        render = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_stream_ledger")
        for node in ast.walk(render):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self.assertFalse(isinstance(target, ast.Attribute) and target.attr == "log")


if __name__ == "__main__":
    unittest.main()
