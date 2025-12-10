"""Utilities for loading SPY intraday and daily datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_INTRADAY_COLUMNS: Iterable[str] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)

REQUIRED_DAILY_COLUMNS: Iterable[str] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_intraday(filepath: str | Path, tz: Optional[str] = "America/New_York") -> pd.DataFrame:
    """
    Load 1-minute OHLCV data and return a time-indexed DataFrame.

    The loader sorts by timestamp and optionally localizes/converts to the
    provided timezone so downstream modules can rely on consistent indexing.
    """

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    _validate_columns(df, REQUIRED_INTRADAY_COLUMNS)

    df = df.sort_values("timestamp").set_index("timestamp")
    if tz:
        df.index = df.index.tz_localize(tz) if df.index.tzinfo is None else df.index.tz_convert(tz)
    return df


def load_daily(filepath: str | Path) -> pd.DataFrame:
    """Load daily OHLCV data for volatility calculations."""

    df = pd.read_csv(filepath, parse_dates=["date"])
    _validate_columns(df, REQUIRED_DAILY_COLUMNS)

    df = df.sort_values("date").set_index("date")
    return df


def resample_to_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Downsample intraday bars to a custom minute interval while preserving volume-weighted prices."""

    if not df.index.inferred_type.startswith("datetime"):
        raise ValueError("DataFrame index must be datetime-like for resampling")

    ohlcv = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(f"{minutes}min").agg(ohlcv).dropna()
