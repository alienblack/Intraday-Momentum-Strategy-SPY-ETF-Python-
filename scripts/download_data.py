#!/usr/bin/env python3
"""
Download SPY intraday and daily data into CSV files.

Default provider: Polygon.io aggregated bars.
Usage example:
  POLYGON_API_KEY=... python scripts/download_data.py --start-date 2019-01-01 --end-date 2024-01-31
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple

import requests


def _iso_from_ms(ms: int) -> str:
    """Convert polygon epoch milliseconds to ISO-8601 with UTC suffix."""
    return dt.datetime.utcfromtimestamp(ms / 1000).isoformat() + "Z"


class PolygonClient:
    def __init__(
        self,
        api_key: str,
        throttle_seconds: float = 0.25,
        base_url: str = "https://api.polygon.io",
        max_retries: int = 5,
        sleep_on_429: float = 5.0,
    ):
        self.api_key = api_key
        self.session = requests.Session()
        self.throttle_seconds = throttle_seconds
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.sleep_on_429 = sleep_on_429

    def fetch_aggs(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        start: str,
        end: str,
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> List[Mapping[str, object]]:
        """Fetch aggregated bars from Polygon, handling pagination."""

        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": str(limit),
            "apiKey": self.api_key,
        }

        results: List[Mapping[str, object]] = []

        while url:
            attempts = 0
            while True:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429 and attempts < self.max_retries:
                    attempts += 1
                    time.sleep(self.sleep_on_429 * attempts)
                    continue
                break

            if resp.status_code != 200:
                raise RuntimeError(f"Polygon error {resp.status_code}: {resp.text}")
            payload = resp.json()
            results.extend(payload.get("results", []))

            next_url = payload.get("next_url")
            # Polygon/Massive sometimes includes apiKey in next_url; avoid duplicating.
            params = None
            if next_url and "apiKey" not in next_url:
                params = {"apiKey": self.api_key}
            url = next_url

            if url:
                time.sleep(self.throttle_seconds)

        return results


def _to_intraday_rows(results: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    rows = []
    for r in results:
        rows.append(
            {
                "timestamp": _iso_from_ms(int(r["t"])),
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            }
        )
    return rows


def _to_daily_rows(results: Iterable[Mapping[str, object]]) -> List[Mapping[str, object]]:
    rows = []
    for r in results:
        ts = dt.datetime.utcfromtimestamp(int(r["t"]) / 1000).date().isoformat()
        rows.append(
            {
                "date": ts,
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"],
            }
        )
    return rows


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SPY intraday (1-min) and daily data to CSV.")
    parser.add_argument("--ticker", default="SPY", help="Ticker to download (default: SPY)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--intraday-path", default="data/spy_1min.csv", help="Output CSV for intraday bars")
    parser.add_argument("--daily-path", default="data/spy_daily.csv", help="Output CSV for daily bars")
    parser.add_argument("--api-key", default=None, help="Polygon API key (or set POLYGON_API_KEY env var)")
    parser.add_argument(
        "--throttle-seconds",
        type=float,
        default=0.25,
        help="Sleep between paginated requests to avoid rate limits (default: 0.25s)",
    )
    parser.add_argument(
        "--base-url",
        default="https://api.polygon.io",
        help="Base URL for API (use https://api.massive.com if using Massive)",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Retries on 429 responses (default: 5)")
    parser.add_argument(
        "--sleep-on-429",
        type=float,
        default=5.0,
        help="Seconds to wait on 429 backoff (multiplied by attempt count).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def month_chunks(start: dt.date, end: dt.date) -> List[Tuple[str, str]]:
    """Return list of (start_iso, end_iso) month-sized chunks inclusive."""
    chunks: List[Tuple[str, str]] = []
    cursor = dt.date(start.year, start.month, 1)
    while cursor <= end:
        next_month = dt.date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
        chunk_end = min(end, next_month - dt.timedelta(days=1))
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = next_month
    return chunks


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    api_key = args.api_key or os.getenv("POLYGON_API_KEY")
    if not api_key:
        print("Missing Polygon API key. Set POLYGON_API_KEY or pass --api-key.", file=sys.stderr)
        return 1

    client = PolygonClient(
        api_key=api_key,
        throttle_seconds=args.throttle_seconds,
        base_url=args.base_url,
        max_retries=args.max_retries,
        sleep_on_429=args.sleep_on_429,
    )

    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    if start_date > end_date:
        print("start-date must be on or before end-date.", file=sys.stderr)
        return 1

    # Chunk intraday calls by month to stay under rate limits.
    intraday_rows: List[Mapping[str, object]] = []
    for chunk_start, chunk_end in month_chunks(start_date, end_date):
        print(f"Fetching intraday 1-min bars for {args.ticker} from {chunk_start} to {chunk_end}...")
        intraday_chunk = client.fetch_aggs(
            args.ticker,
            multiplier=1,
            timespan="minute",
            start=chunk_start,
            end=chunk_end,
        )
        intraday_rows.extend(_to_intraday_rows(intraday_chunk))
        # Brief pause between chunks to avoid per-minute caps.
        time.sleep(args.throttle_seconds)

    _write_csv(Path(args.intraday_path), ["timestamp", "open", "high", "low", "close", "volume"], intraday_rows)
    print(f"Wrote {len(intraday_rows)} intraday rows to {args.intraday_path}")

    print(f"Fetching daily bars for {args.ticker} from {args.start_date} to {args.end_date}...")
    daily = client.fetch_aggs(args.ticker, multiplier=1, timespan="day", start=args.start_date, end=args.end_date)
    daily_rows = _to_daily_rows(daily)
    _write_csv(Path(args.daily_path), ["date", "open", "high", "low", "close", "volume"], daily_rows)
    print(f"Wrote {len(daily_rows)} daily rows to {args.daily_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
