"""Analytics helpers for evaluating the intraday strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


DEFAULT_INTRADAY_PERIODS = 252 * 390  # trading days * minutes per day


def annualized_return(net_returns: pd.Series, periods_per_year: int = DEFAULT_INTRADAY_PERIODS) -> float:
    compounded = (1 + net_returns).prod()
    years = len(net_returns) / periods_per_year
    return compounded ** (1 / max(years, 1e-9)) - 1


def annualized_volatility(net_returns: pd.Series, periods_per_year: int = DEFAULT_INTRADAY_PERIODS) -> float:
    return net_returns.std() * math.sqrt(periods_per_year)


def sharpe_ratio(net_returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = DEFAULT_INTRADAY_PERIODS) -> float:
    excess = net_returns - (rf_rate / periods_per_year)
    vol = annualized_volatility(excess, periods_per_year)
    return excess.mean() / vol if vol != 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


@dataclass
class PerformanceSummary:
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float


def summarize(performance: pd.DataFrame, periods_per_year: int = DEFAULT_INTRADAY_PERIODS) -> PerformanceSummary:
    net_returns = performance["net_return"].dropna()
    equity = performance["equity"].dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0.0
    summary = PerformanceSummary(
        cagr=annualized_return(net_returns, periods_per_year),
        sharpe=sharpe_ratio(net_returns, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity),
        total_return=total_return,
    )
    return summary
