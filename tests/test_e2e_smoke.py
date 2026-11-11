"""First integration smoke test: real CSV -> perception -> forecasting -> verification.

No Ollama, Streamlit, Railway or production writes. This tests the actual
orchestrator and dispatcher; it does NOT certify browser HITL or voice behavior.
"""
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.orchestrator import run_perception
from core.dispatcher import run_capabilities


class EndToEndSmokeTests(unittest.TestCase):
    def test_csv_through_forecast_and_verification(self):
        rng = np.random.default_rng(123)
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        records = []
        for day in dates:
            for customer in range(4):
                qty = 8 + customer + int(day.dayofweek < 5)
                price = 10.0 + customer
                records.append({
                    "transaction_date": day.strftime("%Y-%m-%d"),
                    "customer_id": f"C{customer:03d}",
                    "unit_price": price,
                    "quantity": qty,
                    "revenue": price * qty + float(rng.uniform(0, 0.01)),
                })
        source = io.BytesIO(pd.DataFrame(records).to_csv(index=False).encode())
        with tempfile.TemporaryDirectory() as workdir:
            state = run_perception(
                source, filename="synthetic_revenue.csv",
                nl_request="Forecast revenue for next month",
                base_dir=workdir, use_llm=False,
            )
            self.assertEqual(len(state.raw_df), 360)
            self.assertIsNotNone(state.feature_df)
            self.assertIn("forecasting", state.intents)
            self.assertEqual(state.column_map["date_column"], "transaction_date")
            self.assertEqual(state.column_map["target_column"], "revenue")
            # Restrict only the model roster for CI speed. Backtesting, the
            # actual seasonal-naive runner, Critic and verification remain real.
            from agents import planner_critic
            original = planner_critic.plan_forecasting

            def baseline_plan(s, use_llm=False):
                plan = original(s, use_llm=False)
                plan["candidates"] = ["seasonal_naive"]
                return plan

            with patch.object(planner_critic, "plan_forecasting", side_effect=baseline_plan):
                run_capabilities(state, use_llm=False)

            forecast = state.results.get("forecasting")
            self.assertIsInstance(forecast, dict)
            self.assertNotIn("error", forecast)
            self.assertIn("metrics", forecast)
            self.assertTrue(forecast["metrics"])
            self.assertTrue(any(e.stage == "planner" for e in state.ledger.entries))
            self.assertTrue(any(e.stage == "critic" for e in state.ledger.entries))
            self.assertTrue(any(e.stage == "ingest" for e in state.ledger.entries))
            self.assertTrue(state.ledger.path.is_file())


if __name__ == "__main__":
    unittest.main()
