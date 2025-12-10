# Data directory

No datasets are bundled. Use `scripts/download_data.py` (Polygon by default) to populate:
- `spy_1min.csv`: intraday bars with headers `timestamp,open,high,low,close,volume`
- `spy_daily.csv`: daily bars with headers `date,open,high,low,close,volume`

Keep large CSVs out of git; see `data/.gitignore`.
