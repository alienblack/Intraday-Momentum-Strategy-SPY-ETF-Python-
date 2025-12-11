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


def compute_time_of_day_sigma(intraday: pd.DataFrame, lookback_days: int = 14) -> pd.Series:
    """
    Compute time-of-day noise sigma as the average absolute move from the open over the prior `lookback_days`.

    For each session, the move at HH:MM is |close / session_open - 1|. Sigma for a timestamp is the mean of the
    previous `lookback_days` moves at the same HH:MM (uses only prior sessions).
    """

    df = intraday.copy()
    if not df.index.inferred_type.startswith("datetime"):
        raise ValueError("intraday index must be datetime-like")

    df["session"] = pd.to_datetime(df.index.date)
    df["time"] = df.index.strftime("%H:%M")
    df["session_open"] = df.groupby("session")["open"].transform("first")
    df["move"] = (df["close"] / df["session_open"] - 1.0).abs()

    pivot = df.pivot(index="session", columns="time", values="move").sort_index()
    sigma_pivot = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
    sigma_long = sigma_pivot.stack().rename("sigma").reset_index()

    reset_df = df.reset_index()
    index_col = reset_df.columns[0]
    merged = (
        reset_df.merge(sigma_long, left_on=["session", "time"], right_on=["session", "time"], how="left")
        .set_index(index_col)
    )
    merged.index.name = intraday.index.name or index_col
    return merged["sigma"]


def compute_noise_bands(
    intraday: pd.DataFrame,
    lookback_days: int = 14,
    volatility_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Compute gap-adjusted noise bands per the paper's definition.

    Upper = max(open_t, prev_close) * (1 + VM * sigma_t)
    Lower = min(open_t, prev_close) * (1 - VM * sigma_t)
    where sigma_t is the average absolute move from open to HH:MM over the previous `lookback_days` sessions.
    """

    df = intraday.copy()
    sigma = compute_time_of_day_sigma(df, lookback_days=lookback_days)
    df["session"] = pd.to_datetime(df.index.date)
    df["session_open"] = df.groupby("session")["open"].transform("first")

    session_close = df.groupby("session")["close"].last()
    prev_close_map = session_close.shift()
    df["prev_close"] = df["session"].map(prev_close_map)

    sigma_scaled = sigma * volatility_multiplier
    upper = np.maximum(df["session_open"], df["prev_close"]) * (1 + sigma_scaled)
    lower = np.minimum(df["session_open"], df["prev_close"]) * (1 - sigma_scaled)

    return pd.DataFrame({"sigma": sigma, "upper": upper, "lower": lower}, index=intraday.index)
