"""Simple intraday backtester for the SPY momentum strategy."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .portfolio import Portfolio, PortfolioConfig
from .strategy import IntradayMomentumStrategy, StrategyConfig


def _default_strategy(strategy_config: StrategyConfig | None = None) -> IntradayMomentumStrategy:
    return IntradayMomentumStrategy(strategy_config)


def _default_portfolio(portfolio_config: PortfolioConfig | None = None) -> Portfolio:
    return Portfolio(portfolio_config)


@dataclass
class BacktestResult:
    features: pd.DataFrame
    performance: pd.DataFrame


class Backtester:
    def __init__(
        self,
        strategy: IntradayMomentumStrategy | None = None,
        portfolio: Portfolio | None = None,
    ):
        self.strategy = strategy or _default_strategy()
        self.portfolio = portfolio or _default_portfolio()

    def run(self, intraday: pd.DataFrame) -> BacktestResult:
        """Generate signals, simulate portfolio, and return rich results."""

        features = self.strategy.generate_signals(intraday)
        performance = self.portfolio.simulate(features, positions=features["position"])
        combined_index = features.index.union(performance.index)

        features = features.reindex(combined_index)
        performance = performance.reindex(combined_index)

        return BacktestResult(features=features, performance=performance)
