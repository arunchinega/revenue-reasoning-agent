"""The number-guard must reject LLM prose containing figures absent from
evidence (e.g. leakage restated as '$95,558 revenue decline' plus invented
values), and accept faithful prose with rounded evidence numbers."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import core.quiet  # noqa: F401

from agents.narrator import _llm_numbers_check

EVIDENCE = ('{"leakage": {"total_impact_estimate": 95558.14, "candidate_count": 374},'
            ' "forecasting": {"metrics": [{"model": "arima_auto", "mape": 6.94}]}}')
TEMPLATE = "An estimated 95,558 in revenue is leaking. 374 records. MAPE 6.94%."

good = ("Revenue leakage of 95,558 was found across 374 records; "
        "the best model reached 6.94% error.")
rounded = "Roughly 95,558.1 is at risk — 374 billing records are implicated."
bad_invented = ("Revenue is estimated to be down by 95,558 next month, with "
                "customer CUST0060 contributing a 79,914 decline.")
bad_scaled = "Total exposure is approximately 955,580 across all segments."

assert _llm_numbers_check(good, EVIDENCE, TEMPLATE), "faithful prose rejected"
assert _llm_numbers_check(rounded, EVIDENCE, TEMPLATE), "rounded prose rejected"
assert not _llm_numbers_check(bad_invented, EVIDENCE, TEMPLATE), \
    "invented figure (79,914) accepted"
assert not _llm_numbers_check(bad_scaled, EVIDENCE, TEMPLATE), \
    "scaled figure (955,580) accepted"
print("NARRATION GUARD CHECKS PASSED ✅")
