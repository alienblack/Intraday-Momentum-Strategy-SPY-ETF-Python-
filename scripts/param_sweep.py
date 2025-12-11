#!/usr/bin/env python3
"""
Quick parameter sweep for the intraday momentum strategy on SPY.

Usage:
  python scripts/param_sweep.py --intraday data/spy_1min.csv --daily data/spy_daily.csv --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import summarize_equity  # noqa: E402
from src.backtester import Backtester, BacktesterConfig  # noqa: E402
from src.data_loader import load_daily, load_intraday  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter sweep for VM and sigma_target.")
    p.add_argument("--intraday", default="data/spy_1min.csv")
    p.add_argument("--daily", default="data/spy_daily.csv")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--vm", nargs="+", type=float, default=[0.8, 1.0, 1.2, 1.5], help="Volatility multipliers to test")
    p.add_argument("--sigma-target", nargs="+", type=float, default=[0.015, 0.02, 0.025], help="Target daily vol levels to test")
    p.add_argument("--earliest-entry", default="10:00", help="Earliest HH:MM to begin trading (default: 10:00)")
    p.add_argument("--entry-buffer-pct", type=float, default=0.001, help="Band buffer for entries (default 0.001 = 0.1%)")
    return p.parse_args()


def maybe_clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    idx_tz = df.index.tz
    if start:
        s = pd.to_datetime(start)
        if idx_tz and s.tzinfo is None:
            s = s.tz_localize(idx_tz)
        elif idx_tz and s.tzinfo is not None:
            s = s.tz_convert(idx_tz)
        df = df[df.index >= s]
    if end:
        e = pd.to_datetime(end)
        if idx_tz and e.tzinfo is None:
            e = e.tz_localize(idx_tz)
        elif idx_tz and e.tzinfo is not None:
            e = e.tz_convert(idx_tz)
        df = df[df.index <= e]
    return df


def run_grid(
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    vms: Iterable[float],
    sigma_targets: Iterable[float],
    earliest_entry: str,
    entry_buffer: float,
) -> pd.DataFrame:
    rows = []
    hh, mm = map(int, earliest_entry.split(":"))
    entry_time = pd.Timestamp.today().replace(hour=hh, minute=mm, second=0, microsecond=0).time()
    for vm in vms:
        for sig in sigma_targets:
            cfg = BacktesterConfig(
                volatility_multiplier=vm,
                target_daily_vol=sig,
                earliest_entry_time=entry_time,
                entry_buffer_pct=entry_buffer,
            )
            bt = Backtester(cfg)
            res = bt.run(intraday, daily)
            bench = daily["close"].pct_change().dropna()
            summary = summarize_equity(res.equity, bench)
            rows.append(
                {
                    "vm": vm,
                    "sigma_target": sig,
                    "total_return": summary.total_return,
                    "cagr": summary.cagr,
                    "sharpe": summary.sharpe,
                    "max_dd": summary.max_drawdown,
                    "alpha": summary.alpha,
                    "beta": summary.beta,
                    "hit_ratio": summary.hit_ratio,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    intraday = load_intraday(args.intraday)
    daily = load_daily(args.daily)

    if args.start or args.end:
        intraday = maybe_clip(intraday, args.start, args.end)
        daily = maybe_clip(daily, args.start, args.end)

    df = run_grid(intraday, daily, args.vm, args.sigma_target, args.earliest_entry, args.entry_buffer_pct)
    df = df.sort_values("sharpe", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    best = df.iloc[0]
    print("\nBest by Sharpe:")
    print(best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
