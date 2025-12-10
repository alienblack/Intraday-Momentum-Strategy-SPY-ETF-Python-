"""Intraday momentum strategy logic for SPY."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .noise_area import noise_band, rolling_volatility
from .vwap import intraday_vwap


@dataclass
class StrategyConfig:
    lookback: int = 20
    volatility_multiple: float = 1.0
    max_position: float = 1.0
    transaction_cost_bps: float = 0.5
    hold_bars: int = 30  # approximate intraday holding horizon


def _apply_threshold(momentum: pd.Series, threshold: pd.Series) -> pd.Series:
    signals = pd.Series(0, index=momentum.index, dtype=float)
    signals[momentum > threshold] = 1.0
    signals[momentum < -threshold] = -1.0
    return signals


class IntradayMomentumStrategy:
    """Generate momentum-driven intraday signals with noise filtering."""

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = intraday_vwap(df)
        bands = noise_band(enriched, lookback=self.config.lookback, multiple=self.config.volatility_multiple)
        volatility = rolling_volatility(enriched["close"], lookback=self.config.lookback)

        enriched = enriched.join(bands)
        enriched["volatility"] = volatility
        enriched["momentum"] = enriched["close"].pct_change(self.config.lookback)
        enriched["in_noise"] = enriched["close"].between(enriched["lower"], enriched["upper"])
        return enriched

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce signal and position series.

        - Momentum measured over `lookback` bars.
        - Threshold scales with volatility to avoid trading noise.
        - Positions are capped by `max_position` and held for `hold_bars` unless flipped.
        """

        features = self.build_features(df)

        threshold = features["volatility"] * self.config.volatility_multiple
        raw_signals = _apply_threshold(features["momentum"], threshold)
        raw_signals = raw_signals.where(~features["in_noise"], 0.0)

        positions = raw_signals.replace(0, np.nan).ffill(limit=self.config.hold_bars).fillna(0.0)
        positions = positions.clip(lower=-self.config.max_position, upper=self.config.max_position)

        features["signal"] = raw_signals
        features["position"] = positions
        return features
