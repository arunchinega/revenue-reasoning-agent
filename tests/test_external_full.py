"""End-to-end on the generated external dataset: leakage reconciles to the
manifest, defect recall >= 9/10, ensemble machinery (if chosen) is coherent,
narration renders the winner label."""
import io
import contextlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from agents.narrator import run_narration
from tests.make_external_data import make_external_csv

csv = str(Path(tempfile.gettempdir()) / "utilities_billing.csv")
make_external_csv(csv)
man = json.loads(Path(csv).with_suffix(".manifest.json").read_text())
planted = man["summary"]["total_leakage_impact"]

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="find revenue leakage, forecast next month, "
                        "explain revenue movement, segment customers, recommend actions",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_extfull"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)
    summary = run_narration(st, use_llm=False)

lk, fc = st.results["leakage"], st.results["forecasting"]

# leakage reconciles within 20% of manifest ground truth
assert 0.8 * planted <= lk["total_impact_estimate"] <= 1.2 * planted, \
    f"estimate {lk['total_impact_estimate']:,.0f} vs planted {planted:,.0f}"

# recall: at least 9 of 10 planted leakage customers surface as candidates
leak_cust = {d["customer_id"] for d in man["defects"]
             if d["defect_type"].startswith(("D1", "D2", "D3"))}
found = {c.get("entity") for c in lk["candidates"]}
assert len(leak_cust & found) >= 9, f"recall {len(leak_cust & found)}/10"

# forecast accepted; if ensemble, payload must be complete and label coherent
assert fc["verdict"] in ("accept_single", "accept_ensemble")
assert isinstance(fc["winner_label"], str) and fc["winner_label"] != "—"
if fc["verdict"] == "accept_ensemble":
    e = fc["ensemble"]
    assert e and set(e) >= {"members", "weights", "error_corr", "blend_mape"}
    assert fc["winner_label"].startswith("ensemble(")
    assert abs(sum(e["weights"]) - 1.0) < 1e-6
assert fc["forecast"], "forecast series missing"
if fc["verdict"] == "accept_ensemble":
    assert all(m in summary for m in fc["ensemble"]["members"]), \
        "ensemble members missing from narrative"
    assert "ensemble" in summary.lower()

print(f"planted {planted:,.0f} → estimated {lk['total_impact_estimate']:,.0f} | "
      f"recall {len(leak_cust & found)}/10 | forecast {fc['verdict']} → {fc['winner_label']}")
print("EXTERNAL FULL-RUN CHECKS PASSED ✅")
