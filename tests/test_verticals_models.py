"""Tri-vertical model-zoo proof: utilities, ecommerce, banking — each must
detect its domain, pick the right revenue target, race >=4 models, and land
an ML-family model in the top-2 of at least two verticals."""
import io, contextlib, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401
from core.orchestrator import run_perception
from core.dispatcher import run_capabilities
from tests.make_demo_data import make_utilities_csv
from tests.make_ecommerce_data import make_ecommerce_csv
from tests.make_banking_data import make_banking_csv

tmp = Path(tempfile.gettempdir())
make_utilities_csv(str(tmp / "v_util.csv"))
make_ecommerce_csv(str(tmp / "v_ecom.csv"))
make_banking_csv(str(tmp / "v_bank.csv"))
SPECS = [("v_util.csv", "utilities", "billed_amount"),
         ("v_ecom.csv", "ecommerce", "order_amount"),
         ("v_bank.csv", "banking", "fee_amount")]
ml_top2 = 0
for fname, dom, tgt in SPECS:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        st = run_perception(str(tmp / fname),
                            nl_request="forecast revenue next month and find leakage",
                            base_dir=str(tmp / f"vert_{dom}"), use_llm=False)
        run_capabilities(st, use_llm=False)
    assert st.domain.get("name") == dom, (fname, st.domain.get("name"))
    assert st.column_map.get("target_column") == tgt, \
        (fname, st.column_map.get("target_column"))
    fc = st.results["forecasting"]
    tab = sorted([m for m in fc["metrics"] if m.get("mape") is not None],
                 key=lambda m: m["mape"])
    assert len(tab) >= 4, f"{dom}: only {len(tab)} raced"
    top2fams = {m["family"] for m in tab[:2]}
    if top2fams & {"ml_boosting", "ml_bagging", "deep_learning"}:
        ml_top2 += 1
    print(f"{dom:10s} target={tgt:13s} verdict={fc['verdict']:16s} "
          f"top2={[(m['model'], m['mape']) for m in tab[:2]]}")
assert ml_top2 >= 2, f"ML models in top-2 on only {ml_top2}/3 verticals"
print(f"ML family in top-2 on {ml_top2}/3 verticals")
print("VERTICAL MODEL-ZOO CHECKS PASSED ✅")
