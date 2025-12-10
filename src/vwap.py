"""Volume-weighted average price helpers."""
from __future__ import annotations

import pandas as pd


def intraday_vwap(df: pd.DataFrame, session_label: str = "session") -> pd.DataFrame:
    """
    Compute running VWAP per session and return DataFrame with a `vwap` column.

    Assumes the index is datetime-like and sorted.
    """

    if not df.index.inferred_type.startswith("datetime"):
        raise ValueError("Index must be datetime-like for VWAP computation")

    tmp = df.copy()
    tmp[session_label] = tmp.index.date
    tmp["pv"] = tmp["close"] * tmp["volume"]

    grouped = tmp.groupby(session_label)
    tmp["cum_pv"] = grouped["pv"].cumsum()
    tmp["cum_vol"] = grouped["volume"].cumsum()
    tmp["vwap"] = tmp["cum_pv"] / tmp["cum_vol"]

    return tmp.drop(columns=["pv", "cum_pv", "cum_vol"])


def session_summary(df: pd.DataFrame, session_label: str = "session") -> pd.DataFrame:
    """Return per-session VWAP, OHLC, and volume aggregates."""

    with_vwap = intraday_vwap(df, session_label=session_label)
    grouped = with_vwap.groupby(session_label)

    summary = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vwap=("vwap", "last"),
        volume=("volume", "sum"),
    )
    return summary
