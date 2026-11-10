"""Guard past-only gap handling; a future bill cannot alter earlier gaps."""
import unittest
import pandas as pd
from capabilities.forecasting import prepare_series


class BillingCalendarSafetyTests(unittest.TestCase):
    def test_future_billing_amount_does_not_change_prior_missing_day(self):
        base = pd.DataFrame({
            "bill_date": pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-05"]),
            "billed_amount": [100.0, 400.0, 500.0],
        })
        first = prepare_series(base, "bill_date", "billed_amount", {"fill_gaps": True})
        changed = base.copy()
        changed.loc[1, "billed_amount"] = 40000.0
        second = prepare_series(changed, "bill_date", "billed_amount", {"fill_gaps": True})
        for date in ("2026-01-02", "2026-01-03"):
            self.assertEqual(first.loc[date], 100.0)
            self.assertEqual(second.loc[date], 100.0)

    def test_gap_handling_is_explicitly_opt_in(self):
        df = pd.DataFrame({
            "bill_date": pd.to_datetime(["2026-01-01", "2026-01-04"]),
            "billed_amount": [100.0, 400.0],
        })
        self.assertEqual(len(prepare_series(df, "bill_date", "billed_amount")), 2)
        self.assertEqual(len(prepare_series(df, "bill_date", "billed_amount",
                                            {"fill_gaps": True})), 4)


if __name__ == "__main__":
    unittest.main()
