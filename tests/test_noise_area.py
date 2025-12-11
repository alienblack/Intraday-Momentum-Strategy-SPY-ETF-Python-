import pandas as pd
import pytest

from src.noise_area import compute_noise_bands


def test_noise_band_gap_adjustment_and_sigma():
    """
    Build 15 sessions with a consistent +1% move from open to 10:00.
    On the 15th session, sigma should be 1% (average of prior 14),
    upper should use max(open, prev_close), lower should use min(open, prev_close).
    """

    rows = []
    for i in range(15):
        date = pd.Timestamp("2024-01-{:02d}".format(i + 1))
        open_px = 100 + i  # drift the open so prev_close != open
        # 09:30 bar
        rows.append({"ts": date + pd.Timedelta(hours=9, minutes=30), "open": open_px, "high": open_px, "low": open_px, "close": open_px, "volume": 1000})
        # 10:00 bar (+1%)
        close_10 = open_px * 1.01
        rows.append({"ts": date + pd.Timedelta(hours=10), "open": close_10, "high": close_10, "low": close_10, "close": close_10, "volume": 1000})

    df = pd.DataFrame(rows).set_index("ts")
    df.index = df.index.tz_localize("America/New_York")

    bands = compute_noise_bands(df, lookback_days=14, volatility_multiplier=1.0)

    # Focus on the 10:00 bar of the 15th session
    ts = pd.Timestamp("2024-01-15 10:00", tz="America/New_York")
    row = bands.loc[ts]

    # Sigma should be average of prior 14 days moves = 1%
    assert pytest.approx(0.01, rel=1e-6) == row["sigma"]

    # For day 15: open = 114, prev_close from day 14 = 114.13; max -> 114.13; min -> 114
    expected_upper = 114.13 * 1.01
    expected_lower = 114 * 0.99

    assert pytest.approx(expected_upper, rel=1e-6) == row["upper"]
    assert pytest.approx(expected_lower, rel=1e-6) == row["lower"]
