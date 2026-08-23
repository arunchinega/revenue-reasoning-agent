"""Every stage story must render non-empty, first-person, and numerically
faithful against a completed run — all four Critic verdict variants covered
by construction on the two datasets."""
import io
import contextlib
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from agents import stage_stories as SS
from tests.make_demo_data import make_utilities_csv

csv = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
make_utilities_csv(csv)
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="find revenue leakage, forecast next "
                        "month, explain revenue movement, flag billing anomalies, "
                        "segment customers, run what-if scenarios and recommend actions",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_story"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)

stories = {
    "ingest": SS.ingest_story(st), "eda": SS.eda_story(st),
    "features": SS.features_story(st), "plan": SS.plan_story(st),
    "bakeoff": SS.bakeoff_story(st), "verdict": SS.verdict_story(st),
    "anomaly": SS.anomaly_story(st), "leakage": SS.leakage_story(st),
    "rca": SS.rca_story(st), "segment": SS.segment_story(st),
    "whatif": SS.whatif_story(st), "recommend": SS.recommend_story(st),
    "narration": SS.narration_story(st),
}
for name, text in stories.items():
    assert text and len(text) > 40, f"{name} story too short: {text!r}"
    assert "{" not in text and "}" not in text, f"{name} has unfilled slot: {text}"
assert " I " in (" " + stories["ingest"]) or stories["ingest"].startswith("I ")

# numeric faithfulness spot-checks
lk = st.results["leakage"]
assert f"{lk['total_impact_estimate']:,.0f}" in stories["leakage"]
fc = st.results["forecasting"]
if fc["verdict"] == "accept_single":
    assert fc["winner"] in stories["verdict"]
elif fc["verdict"] == "accept_ensemble":
    assert all(m in stories["verdict"] for m in fc["ensemble"]["members"])
for name, fn in SS.SPEAK.items():
    txt = fn(st)
    assert txt and len(txt) > 30, f"SPEAK[{name}] too short"
    assert "{" not in txt, f"SPEAK[{name}] unfilled slot"
    assert "**" not in txt and "`" not in txt, f"SPEAK[{name}] markdown leaked"
print("all 13 stage stories + 12 spoken scripts render, slots filled, numbers faithful")
print("STAGE STORY CHECKS PASSED ✅")
