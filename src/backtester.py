"""Event-driven intraday backtester for the SPY momentum strategy."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .noise_area import compute_noise_bands
from .vwap import intraday_vwap


RTH_START = dt.time(9, 30)
RTH_END = dt.time(16, 0)


@dataclass
class BacktesterConfig:
    lookback_days: int = 14
    volatility_multiplier: float = 1.0
    target_daily_vol: float = 0.02  # 2%
    max_leverage: float = 4.0
    commission_per_share: float = 0.0035
    slippage_per_share: float = 0.001
    initial_capital: float = 100_000.0
    decision_minutes: Sequence[int] = (0, 30)  # HH:00 and HH:30


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    shares: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df.index.time >= RTH_START) & (df.index.time <= RTH_END)
    return df.loc[mask].copy()


def _add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["session"] = pd.to_datetime(df.index.date)
    df["time"] = df.index.strftime("%H:%M")
    df["session_open"] = df.groupby("session")["open"].transform("first")
    return df


def _prepare_intraday(df: pd.DataFrame, cfg: BacktesterConfig) -> pd.DataFrame:
    filtered = _filter_rth(df).sort_index()
    bands = compute_noise_bands(filtered, lookback_days=cfg.lookback_days, volatility_multiplier=cfg.volatility_multiplier)
    with_bands = filtered.join(bands)
    with_bands = _add_session_columns(with_bands)
    with_vwap = intraday_vwap(with_bands)
    with_vwap["session"] = pd.to_datetime(with_vwap["session"])
    return with_vwap


def _compute_daily_vol(daily: pd.DataFrame, lookback: int = 14) -> pd.Series:
    daily_sorted = daily.sort_index()
    ret = daily_sorted["close"].pct_change()
    vol = ret.rolling(window=lookback, min_periods=lookback).std()
    return vol


class Backtester:
    def __init__(self, config: Optional[BacktesterConfig] = None):
        self.config = config or BacktesterConfig()

    def _calc_shares(self, capital: float, day_open: float, vol: float) -> float:
        if np.isnan(vol) or vol <= 0:
            return 0.0
        leverage = min(self.config.max_leverage, self.config.target_daily_vol / vol)
        return capital * leverage / day_open

    def run(self, intraday: pd.DataFrame, daily: pd.DataFrame) -> BacktestResult:
        df = _prepare_intraday(intraday, self.config)
        daily_vol = _compute_daily_vol(daily)
        daily_vol = daily_vol.shift(1)  # use prior-day vol estimate

        capital = self.config.initial_capital
        trades: List[Trade] = []
        equity_rows: List[dict] = []

        sessions = df["session"].drop_duplicates().sort_values()

        for session in sessions:
            session_df = df[df["session"] == session]
            if session_df.empty:
                continue

            day_open = session_df["session_open"].iloc[0]
            vol_est = daily_vol.get(session, np.nan)
            shares_today = self._calc_shares(capital, day_open, vol_est)
            if shares_today == 0:
                equity_rows.append(
                    {
                        "date": session,
                        "equity_start": capital,
                        "daily_pnl": 0.0,
                        "equity_end": capital,
                        "vol_est": vol_est,
                        "shares": shares_today,
                    }
                )
                continue

            pos = 0  # +1 long, -1 short, 0 flat
            entry_price = None
            entry_time = None
            daily_pnl = 0.0
            session_df = session_df.sort_index()
            last_row = session_df.iloc[-1]

            for ts, row in session_df.iterrows():
                price = row["close"]
                upper = row["upper"]
                lower = row["lower"]
                vwap = row["vwap"]

                if pd.isna(upper) or pd.isna(lower):
                    continue

                decision_time = ts.minute in self.config.decision_minutes and ts.second == 0

                # Trailing stop check at decision times
                if pos > 0 and decision_time:
                    stop = max(upper, vwap)
                    if price <= stop:
                        gross, costs, net = self._close_trade(entry_price, price, shares_today, side="LONG")
                        trades.append(
                            Trade(entry_time, ts, "LONG", shares_today, entry_price, price, gross, costs, net, "stop")
                        )
                        daily_pnl += net
                        pos = 0
                        entry_price = None
                        entry_time = None
                elif pos < 0 and decision_time:
                    stop = min(lower, vwap)
                    if price >= stop:
                        gross, costs, net = self._close_trade(entry_price, price, shares_today, side="SHORT")
                        trades.append(
                            Trade(entry_time, ts, "SHORT", shares_today, entry_price, price, gross, costs, net, "stop")
                        )
                        daily_pnl += net
                        pos = 0
                        entry_price = None
                        entry_time = None

                if not decision_time:
                    continue

                # Entry / flip logic
                desired_pos = pos
                exit_reason = None
                if price > upper:
                    desired_pos = 1
                elif price < lower:
                    desired_pos = -1
                else:
                    desired_pos = 0

                if desired_pos == pos:
                    continue

                # Close existing position if flipping or exiting
                if pos != 0:
                    side = "LONG" if pos > 0 else "SHORT"
                    reason = "reverse" if desired_pos != 0 else "band_exit"
                    gross, costs, net = self._close_trade(entry_price, price, shares_today, side=side)
                    trades.append(
                        Trade(entry_time, ts, side, shares_today, entry_price, price, gross, costs, net, reason)
                    )
                    daily_pnl += net
                    pos = 0
                    entry_price = None
                    entry_time = None

                if desired_pos != 0:
                    pos = desired_pos
                    entry_price = price
                    entry_time = ts

            # End-of-day flat
            if pos != 0 and entry_price is not None:
                side = "LONG" if pos > 0 else "SHORT"
                gross, costs, net = self._close_trade(entry_price, last_row["close"], shares_today, side=side)
                trades.append(
                    Trade(entry_time, last_row.name, side, shares_today, entry_price, last_row["close"], gross, costs, net, "eod")
                )
                daily_pnl += net

            equity_rows.append(
                {
                    "date": session,
                    "equity_start": capital,
                    "daily_pnl": daily_pnl,
                    "equity_end": capital + daily_pnl,
                    "vol_est": vol_est,
                    "shares": shares_today,
                }
            )
            capital += daily_pnl

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_rows).set_index("date")
        return BacktestResult(trades=trades_df, equity=equity_df)

    def _close_trade(self, entry_price: float, exit_price: float, shares: float, side: str) -> tuple[float, float, float]:
        per_share_cost = (self.config.commission_per_share + self.config.slippage_per_share) * 2  # entry + exit
        if side == "LONG":
            gross = (exit_price - entry_price) * shares
        else:
            gross = (entry_price - exit_price) * shares
        costs = per_share_cost * shares
        net = gross - costs
        return gross, costs, net
