# Intraday Momentum Strategy (SPY)

Quick-start scaffolding for experimenting with an intraday momentum strategy on SPY using 1-minute OHLCV data.

## Project layout

```
intraday-momentum-spy/
├─ data/
│  ├─ spy_1min.csv      # raw intraday OHLCV
│  └─ spy_daily.csv     # daily OHLCV for volatility calc
├─ src/
│  ├─ __init__.py
│  ├─ analytics.py
│  ├─ backtester.py
│  ├─ data_loader.py
│  ├─ noise_area.py
│  ├─ portfolio.py
│  ├─ strategy.py
│  └─ vwap.py
├─ notebooks/
│  └─ 01_eda.ipynb      # optional EDA and plotting
├─ config.yaml          # parameters
└─ README.md
```

## Setup

Install Python dependencies (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy
```

## Data acquisition

Intraday and daily SPY data are not bundled. Fetch fresh data (e.g., from Polygon) with the helper script:

```bash
pip install requests
export POLYGON_API_KEY=your_key
python scripts/download_data.py --start-date 2019-01-01 --end-date 2024-01-31 \
  --intraday-path data/spy_1min.csv --daily-path data/spy_daily.csv
```

Adjust date ranges as needed. The script writes `timestamp,open,high,low,close,volume` for intraday and `date,open,high,low,close,volume` for daily bars.

## Usage

1. Drop raw data into `data/spy_1min.csv` and `data/spy_daily.csv` using the headers already provided.
2. Adjust parameters in `config.yaml` (lookback, volatility multiplier, transaction costs, etc.).
3. Run a quick backtest in Python:

```python
import pandas as pd
from src.data_loader import load_intraday
from src.backtester import Backtester

intraday = load_intraday("data/spy_1min.csv")
bt = Backtester()
result = bt.run(intraday)
print(result.performance.tail())
```

## Notes

- Strategy is momentum-based with volatility-scaled thresholds and a noise band derived from ATR.
- Portfolio assumes fractional exposure sizing (±1.0 = 100% notional) and simple bps-based trading costs.
- `notebooks/01_eda.ipynb` is a placeholder for exploratory charts and sanity checks once data is available.
