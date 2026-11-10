"""Tests for forecasting feature availability classifications."""
import unittest

from capabilities.feature_contract import (
    HISTORICAL_ONLY, KNOWN_FUTURE, UNKNOWN,
    classify_feature, select_known_future_features,
)


class FeatureContractTests(unittest.TestCase):
    def test_calendar_features_are_known_in_advance(self):
        for name in ("dow", "month", "is_month_end"):
            self.assertEqual(classify_feature(name, target_column="billed_amount").availability,
                             KNOWN_FUTURE)

    def test_realized_revenue_features_are_not_future_known(self):
        for name in ("billed_amount", "billed_amount_lag1", "billed_amount_rollmean7",
                     "customer_revenue_lag1", "customer_revenue_rollmean7",
                     "expected_revenue", "billed_vs_expected_delta", "rfm_monetary"):
            self.assertEqual(classify_feature(name, target_column="billed_amount").availability,
                             HISTORICAL_ONLY, name)

    def test_unknown_columns_fail_closed(self):
        self.assertEqual(classify_feature("future_price", target_column="revenue").availability,
                         UNKNOWN)

    def test_explicit_future_known_requires_assertion(self):
        columns = ["dow", "future_price", "expected_revenue", "unknown"]
        self.assertEqual(select_known_future_features(columns, target_column="revenue"),
                         ["dow"])
        self.assertEqual(select_known_future_features(columns, target_column="revenue",
                                                       future_known=["future_price"]),
                         ["dow", "future_price"])


if __name__ == "__main__":
    unittest.main()
