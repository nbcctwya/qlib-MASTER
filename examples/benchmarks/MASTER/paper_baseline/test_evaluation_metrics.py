import math
import unittest

import numpy as np

from evaluation_metrics import portfolio_metrics


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


if __name__ == "__main__":
    unittest.main()
