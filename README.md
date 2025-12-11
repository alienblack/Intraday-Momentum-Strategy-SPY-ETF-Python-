# Intraday Momentum Strategy (SPY)

Event-driven intraday momentum backtester on 1-minute SPY OHLCV with dynamic “noise bands”, VWAP trailing stops, and volatility-targeted sizing. Includes trade logs, equity output, and basic analytics (Sharpe, max drawdown, alpha/beta vs SPY, monthly returns).

## Project layout

```
intraday-momentum-spy/
├─ data/
│  ├─ spy_1min.csv      # intraday OHLCV (not tracked)
│  └─ spy_daily.csv     # daily OHLCV (not tracked)
├─ src/
│  ├─ analytics.py      # Sharpe, max DD, alpha/beta, monthly returns
│  ├─ backtester.py     # event-driven engine (bands, VWAP stop, sizing)
│  ├─ data_loader.py    # CSV loaders
│  ├─ noise_area.py     # time-of-day sigma, gap-adjusted bands
│  ├─ vwap.py           # session VWAP
│  └─ ...
├─ scripts/
│  ├─ download_data.py  # fetch intraday/daily CSVs (Polygon/Massive)
│  ├─ run_backtest.py   # run strategy and write trades/equity
│  └─ param_sweep.py    # optional parameter sweep (VM, sizing, entry rules)
├─ tests/               # minimal unit tests for bands and VWAP
└─ README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy
```

## Data

Data is not bundled. Fetch from Polygon/Massive (adjust dates):
```bash
pip install requests
export POLYGON_API_KEY=your_key
python scripts/download_data.py --base-url https://api.massive.com \
  --start-date 2024-01-01 --end-date 2024-12-31 \
  --intraday-path data/spy_1min.csv --daily-path data/spy_daily.csv
```
Files should have headers:
- `data/spy_1min.csv`: timestamp,open,high,low,close,volume
- `data/spy_daily.csv`: date,open,high,low,close,volume

Keep CSVs untracked (separate data branch recommended).

## Run backtest

```bash
python scripts/run_backtest.py \
  --start 2024-01-01 --end 2024-12-31 \
  --intraday data/spy_1min.csv --daily data/spy_daily.csv \
  --output-dir outputs \
  --earliest-entry 10:00 \
  --entry-buffer-pct 0.001   # 0.1% band buffer
```

Outputs:
- `outputs/trades.csv`: entry/exit, side, price, size, PnL, exit reason.
- `outputs/equity.csv`: daily equity_start/end, PnL, vol estimate, shares.

Console summary: total return, CAGR, Sharpe, max DD, alpha/beta vs SPY, hit ratio, monthly returns.

## Parameter sweep (optional)

Test grids of volatility multiplier (VM), sizing target, entry time, and band buffer:
```bash
python scripts/param_sweep.py \
  --start 2024-01-01 --end 2024-12-31 \
  --intraday data/spy_1min.csv --daily data/spy_daily.csv
```
Defaults are narrowed for runtime; adjust flags for broader sweeps.

## Notes and limitations

- Follows the paper’s rules: time-of-day noise bands with gap adjustment, semi-hourly decisions, VWAP trailing stop, vol-target sizing (σ_target with 4x cap), costs of $0.0035 commission + $0.001 slippage per share.
- Recent runs on 2024 data show weak Sharpe; treat results as research, not production-ready alpha.
- Keep data CSVs untracked; use a separate data branch if needed.

## CI

GitHub Actions runs the unit tests (bands and VWAP) on pushes/PRs to `main` and `dev`.
