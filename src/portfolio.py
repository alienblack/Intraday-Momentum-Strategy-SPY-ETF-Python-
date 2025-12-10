"""Portfolio mechanics for applying signals to SPY intraday prices."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PortfolioConfig:
    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 0.5


class Portfolio:
    def __init__(self, config: PortfolioConfig | None = None):
        self.config = config or PortfolioConfig()

    def simulate(self, prices: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
        """
        Apply positions to price returns and compute equity curve.

        Positions are interpreted as fractional exposures of capital (e.g., 1.0 = 100% long).
        """

        returns = prices["close"].pct_change().fillna(0.0)
        pos = positions.fillna(0.0)

        turnover = pos.diff().abs().fillna(pos.abs())
        trading_cost = turnover * (self.config.transaction_cost_bps / 10_000)

        strategy_return = pos.shift().fillna(0.0) * returns
        net_return = strategy_return - trading_cost

        equity = (1 + net_return).cumprod() * self.config.initial_capital

        results = pd.DataFrame(
            {
                "returns": returns,
                "position": pos,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "strategy_return": strategy_return,
                "net_return": net_return,
                "equity": equity,
            }
        )
        return results
