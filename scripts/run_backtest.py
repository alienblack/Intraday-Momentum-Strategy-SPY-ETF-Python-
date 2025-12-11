#!/usr/bin/env python3
"""
Run the intraday momentum backtest on SPY data and print summary stats.

Example:
  python scripts/run_backtest.py \
    --intraday data/spy_1min.csv \
    --daily data/spy_daily.csv \
    --start 2024-01-01 --end 2024-12-31 \
    --output-dir outputs
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import sys

import pandas as pd

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import monthly_returns, summarize_equity  # noqa: E402
from src.backtester import Backtester, BacktesterConfig  # noqa: E402
from src.data_loader import load_daily, load_intraday  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SPY intraday momentum backtest.")
    parser.add_argument("--intraday", default="data/spy_1min.csv", help="Path to 1-min intraday CSV")
    parser.add_argument("--daily", default="data/spy_daily.csv", help="Path to daily CSV")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--output-dir", default="outputs", help="Directory to write trades/equity CSVs")
    parser.add_argument("--base-url", default=None, help="Ignored (kept for CLI compatibility with download script).")
    parser.add_argument("--earliest-entry", default="10:00", help="Earliest HH:MM to begin trading (default: 10:00)")
    parser.add_argument("--entry-buffer-pct", type=float, default=0.001, help="Require price to clear band by this pct before entry (default 0.001 = 0.1%)")
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    intraday = load_intraday(args.intraday)
    daily = load_daily(args.daily)

    if args.start or args.end:
        intraday = maybe_clip(intraday, args.start, args.end)
        daily = maybe_clip(daily, args.start, args.end)

    hh, mm = map(int, args.earliest_entry.split(":"))
    cfg = BacktesterConfig(
        earliest_entry_time=dt.time(hh, mm),
        entry_buffer_pct=args.entry_buffer_pct,
    )
    bt = Backtester(cfg)
    result = bt.run(intraday, daily)

    # Benchmark: daily SPY returns
    benchmark_returns = daily["close"].pct_change().dropna()

    summary = summarize_equity(result.equity, benchmark_returns)
    mret = monthly_returns(result.equity["equity_end"] / result.equity["equity_start"] - 1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity.csv"

    result.trades.to_csv(trades_path, index=False)
    result.equity.to_csv(equity_path)

    print("=== Performance Summary ===")
    print(f"Total return: {summary.total_return:.2%}")
    print(f"CAGR:         {summary.cagr:.2%}")
    print(f"Sharpe:       {summary.sharpe:.2f}")
    print(f"Max DD:       {summary.max_drawdown:.2%}")
    print(f"Alpha:        {summary.alpha:.2%}")
    print(f"Beta:         {summary.beta:.2f}")
    print(f"Hit ratio:    {summary.hit_ratio:.2%}")
    print()
    print("=== Monthly Returns (last 12) ===")
    print(mret.tail(12).to_string(float_format=lambda x: f"{x:.2%}"))  # type: ignore[arg-type]
    print()
    print(f"Trades saved to:  {trades_path}")
    print(f"Equity saved to:  {equity_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
