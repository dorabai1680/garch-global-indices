from datetime import date
import unittest

from src.garch_indices.config import DEFAULT_INDICES
import pandas as pd

from src.garch_indices.data import calculate_log_returns, generate_demo_prices, _normalize_price_frame
from src.garch_indices.garch import fit_garch11
from src.garch_indices.forecast_eval import evaluate_frame
from src.garch_indices.reporting import build_summary


class GarchWorkflowTest(unittest.TestCase):
    def test_demo_data_can_be_fitted_and_summarized(self):
        prices = generate_demo_prices(DEFAULT_INDICES, date(2020, 1, 1), date(2020, 12, 31))
        results = {}
        for name, frame in prices.items():
            returns = calculate_log_returns(frame)
            result = fit_garch11(returns["return_pct"].to_numpy(), max_iterations=120)
            self.assertGreater(result.omega, 0)
            self.assertGreaterEqual(result.alpha, 0)
            self.assertGreaterEqual(result.beta, 0)
            self.assertLess(result.persistence, 1)
            results[name] = result
        summary = build_summary(results)
        self.assertEqual(len(summary), len(DEFAULT_INDICES))
        self.assertIn("persistence_alpha_plus_beta", summary.columns)

    def test_rejects_too_short_demo_period(self):
        with self.assertRaises(ValueError):
            generate_demo_prices(DEFAULT_INDICES, date(2020, 1, 1), date(2020, 1, 31))

    def test_forecast_evaluation_has_all_methods_and_metrics(self):
        prices = generate_demo_prices(DEFAULT_INDICES[:1], date(2020, 1, 1), date(2021, 6, 30))
        returns = calculate_log_returns(prices["S&P 500"])
        evaluation = evaluate_frame(returns, refit_every=40, max_iterations=80)
        self.assertEqual(set(evaluation["method"]), {"Historical", "EWMA", "GARCH(1,1)"})
        self.assertTrue({"RMSE", "MAE", "QLIKE"}.issubset(evaluation.columns))
        self.assertFalse(evaluation[["RMSE", "MAE", "QLIKE"]].isna().any().any())

    def test_raw_yahoo_columns_are_normalized(self):
        raw = pd.DataFrame({"Date": ["2024-01-02"], "Adj Close": [100.5]})
        result = _normalize_price_frame(raw, DEFAULT_INDICES[0])
        self.assertEqual(result.columns.tolist(), ["date", "index", "ticker", "adj_close"])
        self.assertEqual(result.loc[0, "adj_close"], 100.5)

    def test_forecast_evaluation_rejects_invalid_decay(self):
        prices = generate_demo_prices(DEFAULT_INDICES[:1], date(2020, 1, 1), date(2021, 1, 1))
        returns = calculate_log_returns(prices["S&P 500"])
        with self.assertRaises(ValueError):
            evaluate_frame(returns, ewma_lambda=1.0)


if __name__ == "__main__":
    unittest.main()
