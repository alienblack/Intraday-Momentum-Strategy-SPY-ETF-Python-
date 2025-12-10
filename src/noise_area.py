"""Volatility and noise-band helpers for intraday signals."""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Compute True Range used by ATR."""

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    return ranges.max(axis=1)


def average_true_range(df: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Smoothed ATR for noise estimation."""

    tr = true_range(df)
    return tr.rolling(window=lookback, min_periods=lookback).mean()


def rolling_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Rolling standard deviation of returns as a fast volatility proxy."""

    returns = close.pct_change()
    vol = returns.rolling(window=lookback, min_periods=lookback).std()
    return vol * np.sqrt(252)  # annualized approximation


def noise_band(df: pd.DataFrame, lookback: int = 20, multiple: float = 1.0) -> pd.DataFrame:
    """
    Build a simple noise area using ATR around a rolling mean.

    The band can filter out choppy periods: prices outside the band are
    considered to have escaped recent noise.
    """

    atr = average_true_range(df, lookback)
    basis = df["close"].rolling(window=lookback, min_periods=lookback).mean()

    upper = basis + multiple * atr
    lower = basis - multiple * atr
    return pd.DataFrame({"basis": basis, "upper": upper, "lower": lower})
