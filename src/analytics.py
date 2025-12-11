"""Analytics helpers for evaluating the intraday strategy."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_PERIODS_PER_YEAR = 252  # daily observations


def daily_returns_from_equity(equity: pd.DataFrame) -> pd.Series:
    """Compute daily returns from equity_start/equity_end columns."""
    ret = equity["equity_end"] / equity["equity_start"] - 1
    return ret.dropna()


def annualized_return(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    compounded = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return compounded ** (1 / max(years, 1e-9)) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    return returns.std() * math.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR) -> float:
    excess = returns - (rf_rate / periods_per_year)
    vol = annualized_volatility(excess, periods_per_year)
    return excess.mean() / vol if vol != 0 else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    rf_rate: float = 0.0,
) -> Tuple[float, float]:
    """Compute CAPM alpha/beta vs benchmark using simple linear regression."""
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty or aligned.iloc[:, 1].var() == 0:
        return 0.0, 0.0

    strat = aligned.iloc[:, 0] - (rf_rate / periods_per_year)
    bench = aligned.iloc[:, 1] - (rf_rate / periods_per_year)

    beta = strat.cov(bench) / bench.var()
    alpha_daily = strat.mean() - beta * bench.mean()
    alpha_annual = alpha_daily * periods_per_year
    return alpha_annual, beta


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns into monthly returns."""
    return returns.add(1).resample("M").prod().sub(1)


def hit_ratio(returns: pd.Series) -> float:
    return (returns > 0).mean() if len(returns) else 0.0


@dataclass
class PerformanceSummary:
    cagr: float
    sharpe: float
    max_drawdown: float
    total_return: float
    alpha: float
    beta: float
    hit_ratio: float


def summarize_equity(
    equity: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    rf_rate: float = 0.0,
) -> PerformanceSummary:
    returns = daily_returns_from_equity(equity)
    equity_curve = equity["equity_end"]

    alpha_val, beta_val = (0.0, 0.0)
    if benchmark_returns is not None:
        alpha_val, beta_val = alpha_beta(returns, benchmark_returns, periods_per_year, rf_rate)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1 if len(equity_curve) > 1 else 0.0
    summary = PerformanceSummary(
        cagr=annualized_return(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, rf_rate=rf_rate, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        total_return=total_return,
        alpha=alpha_val,
        beta=beta_val,
        hit_ratio=hit_ratio(returns),
    )
    return summary
