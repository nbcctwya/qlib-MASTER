import math
import unittest

import numpy as np

from evaluation_metrics import portfolio_metrics
from export_protocol_results import STANDARD_EXECUTOR, STANDARD_STRATEGY, standard_backtest_config


class PortfolioMetricsTest(unittest.TestCase):
    def test_first_day_loss_sets_drawdown(self):
        self.assertAlmostEqual(portfolio_metrics([-0.10])["MDD"], -0.10)

    def test_identical_negative_returns_have_defined_sortino(self):
        result = portfolio_metrics([-0.01, -0.01])
        self.assertTrue(math.isfinite(result["Sortino"]))
        self.assertAlmostEqual(result["Sortino"], -math.sqrt(252))

    def test_total_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            portfolio_metrics([0.01, -1.0])

    def test_independent_manual_calculation(self):
        returns = np.array([0.01, -0.02, 0.03, -0.01])
        logs = np.log(1 + returns)
        nav = np.r_[1.0, np.exp(np.cumsum(logs))]
        expected = {
            "AR": np.exp(logs.mean() * 252) - 1,
            "STD": logs.std(ddof=1) * np.sqrt(252),
            "MDD": np.min(nav / np.maximum.accumulate(nav) - 1),
            "Sharpe": np.sqrt(252) * logs.mean() / logs.std(ddof=1),
            "Sortino": np.sqrt(252) * logs.mean() / np.sqrt(np.mean(np.minimum(logs, 0) ** 2)),
        }
        expected["Calmar"] = expected["AR"] / abs(expected["MDD"])
        actual = portfolio_metrics(returns)
        for metric, value in expected.items():
            self.assertAlmostEqual(actual[metric], value)

    def test_standard_backtest_parameters_are_explicit(self):
        config = {
            "market": "csi300",
            "benchmark": "SH000300",
            "task": {"dataset": {"kwargs": {"segments": {"test": ["2023-01-01", "2025-12-31"]}}}},
            "port_analysis_config": {"backtest": {"exchange_kwargs": {"limit_threshold": 0.095, "deal_price": "close"}}},
        }
        import pandas as pd

        index = pd.MultiIndex.from_tuples([(pd.Timestamp("2023-01-03"), "SH600000")], names=["datetime", "instrument"])
        backtest = standard_backtest_config(pd.Series([1.0], index=index), config)
        self.assertEqual(backtest["strategy"]["kwargs"] | {"signal": None}, STANDARD_STRATEGY["kwargs"] | {"signal": None})
        self.assertEqual(backtest["executor"], STANDARD_EXECUTOR)
        self.assertEqual(backtest["account"], 100000000)
        self.assertEqual(backtest["start_time"], "2023-01-01")
        self.assertEqual(backtest["end_time"], "2025-12-31")
        self.assertEqual(backtest["exchange_kwargs"]["freq"], "day")
        self.assertEqual(backtest["exchange_kwargs"]["codes"], "csi300")
        self.assertEqual(backtest["exchange_kwargs"]["open_cost"], 0.0005)
        self.assertEqual(backtest["exchange_kwargs"]["close_cost"], 0.0015)
        self.assertEqual(backtest["exchange_kwargs"]["min_cost"], 0)


if __name__ == "__main__":
    unittest.main()
