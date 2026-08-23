"""Parallel dispatch must produce the same result set as sequential, fire the
callback once per capability, and isolate failures."""
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
from core.dispatcher import run_capabilities_parallel
from tests.make_demo_data import make_utilities_csv

csv = str(Path(tempfile.gettempdir()) / "demo_utilities.csv")
make_utilities_csv(csv)
NL = ("find revenue leakage, forecast next month, explain why revenue moved, "
      "flag billing anomalies, segment customers, run what-if scenarios and "
      "recommend actions")
done = []
buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    st = run_perception(csv, nl_request=NL,
                        base_dir=str(Path(tempfile.gettempdir()) / "rra_par"),
                        use_llm=False)
    run_capabilities_parallel(st, use_llm=False,
                              on_capability_done=lambda i, s: done.append(i))

expected = {"segment", "anomaly", "leakage", "forecasting", "rca", "whatif",
            "recommend"}
assert set(done) == expected, f"callback set mismatch: {sorted(done)}"
assert len(done) == len(expected), "callback fired more than once per capability"
for cap in expected:
    assert cap in st.results, f"{cap} missing from results"
    assert "error" not in st.results[cap], f"{cap} errored: {st.results[cap]}"
# dependents ran after their inputs existed
assert done.index("rca") > max(done.index(x) for x in ("anomaly", "leakage", "segment"))
assert done.index("whatif") > done.index("forecasting")
fc = st.results["forecasting"]
assert fc["verdict"] in ("accept_single", "accept_ensemble")
print(f"parallel order: {done} | forecast {fc['verdict']} → {fc.get('winner_label')}")
print("PARALLEL DISPATCH CHECKS PASSED ✅")
