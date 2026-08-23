"""Second-domain regression: ecommerce (generic profile) — discount-aware
expectations, excessive-discount rule, and manifest reconciliation."""
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
from tests.make_ecommerce_data import make_ecommerce_csv

csv = str(Path(tempfile.gettempdir()) / "ecommerce_india.csv")
make_ecommerce_csv(csv)
man = json.loads(Path(csv).with_suffix(".manifest.json").read_text())
planted = man["summary"]["total_leakage_impact"] + \
    man["summary"]["by_type"].get("E5_duplicate_orders", 0)

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="find revenue leakage, forecast next "
                        "month, explain why revenue moved, flag order anomalies, "
                        "segment customers and recommend actions",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_ecom"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)

lk, fc = st.results["leakage"], st.results["forecasting"]
assert 0.75 * planted <= lk["total_impact_estimate"] <= 1.25 * planted, \
    f"estimate {lk['total_impact_estimate']:,.0f} vs planted {planted:,.0f}"
found = {c.get("entity") for c in lk["candidates"]}
leak_cust = {d["customer_id"] for d in man["defects"]
             if d["defect_type"].startswith(("E1", "E2", "E3"))}
assert len(leak_cust & found) >= 10, f"recall {len(leak_cust & found)}/11"
fired = {k for k, v in lk["rules_fired"].items() if v.get("hits")}
assert "excessive_discount" in fired, "discount-abuse rule silent"
assert fc["verdict"] in ("accept_single", "accept_ensemble"), \
    "contrast dataset should certify"
hyp = (st.results.get("rca", {}).get("hypotheses") or [{}])[0].get("hypothesis", "")
assert "Telangana" in hyp, f"regional dip missed: {hyp}"
print(f"planted+dupes {planted:,.0f} → est {lk['total_impact_estimate']:,.0f} | "
      f"recall {len(leak_cust & found)}/11 | {fc['verdict']} → "
      f"{fc.get('winner_label')} | RCA found Telangana")
print("ECOMMERCE CHECKS PASSED ✅")
