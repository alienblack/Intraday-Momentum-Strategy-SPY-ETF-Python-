import pandas as pd

from src.vwap import intraday_vwap


def test_intraday_vwap_per_session():
    """
    Two sessions, simple prices/volumes; VWAP should be cumulative per day only.
    """

    rows = [
        # Day 1
        {"ts": "2024-01-02 09:30", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 100},
        {"ts": "2024-01-02 09:31", "open": 100, "high": 102, "low": 99, "close": 102, "volume": 200},
        # Day 2
        {"ts": "2024-01-03 09:30", "open": 103, "high": 104, "low": 102, "close": 103, "volume": 100},
        {"ts": "2024-01-03 09:31", "open": 103, "high": 105, "low": 103, "close": 104, "volume": 300},
    ]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts")

    out = intraday_vwap(df)

    day1_last = out.loc[pd.Timestamp("2024-01-02 09:31", tz="UTC"), "vwap"]
    # Day1 VWAP = (100*100 + 102*200)/(100+200) = (10000 + 20400)/300 = 101.333...
    assert abs(day1_last - 101.3333333) < 1e-6

    day2_last = out.loc[pd.Timestamp("2024-01-03 09:31", tz="UTC"), "vwap"]
    # Day2 VWAP = (103*100 + 104*300)/(100+300) = (10300 + 31200)/400 = 103.75
    assert abs(day2_last - 103.75) < 1e-6
