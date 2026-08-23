"""Regression: the system must survive every realistic LLM output shape —
the audit findings that crashed Arun's first live-fire run."""
import io
import contextlib
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from core.llm import as_text, as_float, as_str_list
from core.orchestrator import run_perception
from agents import planner_critic as PC
from tests.make_demo_data import make_utilities_csv

# unit: coercers
assert as_text(["a", "b"]) == "a → b"
assert as_text({"s": 1}) == "s: 1"
assert as_float("8%") == 8.0
assert as_float(None, 5.0) == 5.0
assert as_str_list([{"model": "ets"}, 42, "naive"]) == ["ets", "naive"]

csv = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
make_utilities_csv(csv)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="forecast revenue",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_shapes"),
                        use_llm=False)

CASES = [
    {"candidates": ["seasonal_naive", "ets", "xgboost_lags"],
     "reasoning": ["step 1", "step 2"], "preprocessing": {"fill_gaps": True},
     "acceptance_mape": 8.0},
    {"candidates": [{"model": "ets"}, "seasonal_naive"], "reasoning": "ok",
     "preprocessing": ["winsorize"], "acceptance_mape": "8%"},
    {"candidates": [{"x": "prophet"}, 42, "ets"], "reasoning": {"steps": "eh"},
     "preprocessing": None, "acceptance_mape": None},
]


class F:
    def __init__(self, parsed):
        self.parsed, self.used_fallback = parsed, False


for i, parsed in enumerate(CASES):
    st.plan = {}
    with patch.object(PC, "call_json", lambda **kw: F(dict(CASES[0] if False else parsed))):
        with contextlib.redirect_stdout(io.StringIO()):
            plan = PC.plan_forecasting(st, use_llm=True)
    assert isinstance(plan["reasoning"], str)
    assert all(isinstance(c, str) for c in plan["candidates"])
    assert isinstance(plan["acceptance_mape"], float)
    assert "seasonal_naive" in plan["candidates"]

# intent floor: LLM shrinkage restored from keyword evidence
from stages import intent_detect as ID
with patch.object(ID, "call_json",
                  lambda **kw: F({"intents": ["leakage"], "confidence": 0.9,
                                  "reasoning": "narrow read"})):
    st.nl_request = ("find revenue leakage, forecast next month, flag billing "
                     "anomalies and segment customers")
    with contextlib.redirect_stdout(io.StringIO()):
        ID.run_intent_detection(st, use_llm=True)
floor = set(st.intents)
assert {"leakage", "forecasting", "anomaly", "segment"} <= floor, \
    f"intent floor failed: {st.intents}"

print("coercers OK | 3 adversarial plans OK | intent floor restores dropped intents")
print("LLM SHAPE CHECKS PASSED ✅")
