"""Regression: external-style CSV (string tariff_code) must not crash the
feature engine, must trigger the inferred-price rule, and must recover the
planted unbilled window with a realistic impact estimate."""
import io
import contextlib
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

import numpy as np
import pandas as pd

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities

rng = np.random.default_rng(42)
days = pd.date_range("2025-03-01", periods=540, freq="D")
RATES = {"RES-STD": 6.5, "COM-STD": 8.2, "IND-STD": 7.1}
FIXED = {"residential": 120, "commercial": 450, "industrial": 2000}
rows = []
planted_impact = 0.0
for cid in range(40):
    seg = "residential" if cid < 28 else ("commercial" if cid < 37 else "industrial")
    tc = {"residential": "RES-STD", "commercial": "COM-STD", "industrial": "IND-STD"}[seg]
    base = {"residential": 30, "commercial": 120, "industrial": 800}[seg]
    for d in days:
        kwh = base * (1 + 0.1 * np.sin(d.dayofweek)) * rng.normal(1, 0.06)
        billed = kwh * RATES[tc] + FIXED[seg] / 30
        if cid == 5 and pd.Timestamp("2025-10-01") <= d <= pd.Timestamp("2025-10-14"):
            planted_impact += billed
            billed = 0.0
        rows.append((f"CUST{cid:04d}", d, seg, tc, round(kwh, 2), round(billed, 2)))
df = pd.DataFrame(rows, columns=["customer_id", "bill_date", "segment",
                                 "tariff_code", "kwh_consumed", "billed_amount"])
csv = str(Path(tempfile.gettempdir()) / "external_style.csv")
df.to_csv(csv, index=False)

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request="find leakage and anomalies and forecast",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_ext"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)

fr = st.feature_report["applied_rules"]
assert "price_qty" not in fr, "string tariff_code must NOT match as numeric price"
assert "inferred_price" in fr, f"inferred-price rule should fire, got {list(fr)}"

lk = st.results["leakage"]
c5 = [c for c in lk["candidates"] if "CUST0005" in str(c.get("entity", ""))]
assert c5, "planted unbilled customer missing from candidates"
impact = lk["total_impact_estimate"]
assert 0.75 * planted_impact < impact < 1.25 * planted_impact, (
    f"impact {impact:,.0f} should be within 25% of planted {planted_impact:,.0f}")
assert st.results["forecasting"]["verdict"] in ("accept_single", "accept_ensemble")

print(f"rules: {list(fr)} | planted {planted_impact:,.0f} → estimated {impact:,.0f} "
      f"| CUST0005 found: True | forecast: {st.results['forecasting']['winner']}")
print("EXTERNAL SCHEMA CHECKS PASSED ✅")
