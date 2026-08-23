"""Width stress: ecommerce base + 11 extra realistic columns (~20 total).
Reconciliation and recall must hold — proves column count is a non-issue."""
import io
import contextlib
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from tests.make_ecommerce_data import make_ecommerce_csv

base = str(Path(tempfile.gettempdir()) / "ecom_wide_base.csv")
make_ecommerce_csv(base)
man = json.loads(Path(base).with_suffix(".manifest.json").read_text())
planted = man["summary"]["total_leakage_impact"] + \
    man["summary"]["by_type"].get("E5_duplicate_orders", 0)

df = pd.read_csv(base)
rng = np.random.default_rng(11)
n = len(df)
df["payment_method"] = rng.choice(["upi", "card", "cod", "netbanking"], n)
df["channel"] = rng.choice(["app", "web", "marketplace"], n, p=[.55, .3, .15])
df["courier"] = rng.choice(["Delhivery", "BlueDart", "Ekart", "DTDC"], n)
df["city_tier"] = rng.choice(["tier1", "tier2", "tier3"], n, p=[.5, .3, .2])
df["customer_rating"] = rng.integers(1, 6, n).astype(float)
df.loc[rng.choice(n, n // 50, replace=False), "customer_rating"] = np.nan
df["is_return"] = rng.choice([0, 1], n, p=[.94, .06])
df["tax_amount"] = (df["order_amount"] * 0.18).round(2)
df["shipping_fee"] = rng.choice([0, 40, 79, 129], n).astype(float)
df["coupon_code"] = rng.choice(["", "SAVE10", "FEST20", "NEW50"], n,
                               p=[.7, .12, .12, .06])
df["device"] = rng.choice(["android", "ios", "desktop"], n, p=[.6, .25, .15])
df["session_minutes"] = rng.exponential(9, n).round(1)
wide = str(Path(tempfile.gettempdir()) / "ecom_wide.csv")
df.to_csv(wide, index=False)

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(wide, nl_request="find revenue leakage, forecast next "
                        "month, explain why revenue moved, flag order anomalies "
                        "and recommend actions",
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_wide"),
                        use_llm=False)
    run_capabilities(st, use_llm=False)

assert len(df.columns) >= 20, f"only {len(df.columns)} cols"
assert st.domain.get("name") == "ecommerce", st.domain.get("name")
lk = st.results["leakage"]
found = {c.get("entity") for c in lk["candidates"]}
leak_cust = {d["customer_id"] for d in man["defects"]
             if d["defect_type"].startswith(("E1", "E2", "E3"))}
assert len(leak_cust & found) >= 10, f"recall {len(leak_cust & found)}/11"
assert 0.7 * planted <= lk["total_impact_estimate"] <= 1.3 * planted, \
    f"{lk['total_impact_estimate']:,.0f} vs {planted:,.0f}"
fc = st.results["forecasting"]
assert fc["verdict"] in ("accept_single", "accept_ensemble")
print(f"{len(df.columns)} columns | est {lk['total_impact_estimate']:,.0f} "
      f"vs planted {planted:,.0f} "
      f"({lk['total_impact_estimate']/planted:.0%}) | recall "
      f"{len(leak_cust & found)}/11 | {fc['verdict']}")
print("WIDE SCHEMA CHECKS PASSED ✅")
