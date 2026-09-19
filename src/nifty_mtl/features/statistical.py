"""Statistical features on weekly returns (10 features, PRD §7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATISTICAL_NAMES = [
    "ret_1w", "ret_4w", "ret_12w", "ret_26w", "vol_4w", "vol_12w",
    "skew_12w", "kurt_12w", "acf1_12w", "volume_ratio_12w",
]


def _rolling_acf1(ret: pd.DataFrame, n: int) -> pd.DataFrame:
    """Lag-1 autocorrelation over a trailing window, per column."""
    lag = ret.shift(1)
    x = ret - ret.rolling(n).mean()
    y = lag - lag.rolling(n).mean()
    cov = (x * y).rolling(n).mean()
    return cov / (ret.rolling(n).std(ddof=0) * lag.rolling(n).std(ddof=0)).replace(0, np.nan)


def statistical_features(ret: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feats = {
        "ret_1w": ret,
        "ret_4w": ret.rolling(4).sum(),
        "ret_12w": ret.rolling(12).sum(),
        "ret_26w": ret.rolling(26).sum(),
        "vol_4w": ret.rolling(4).std(),
        "vol_12w": ret.rolling(12).std(),
        "skew_12w": ret.rolling(12).skew(),
        "kurt_12w": ret.rolling(12).kurt(),
        "acf1_12w": _rolling_acf1(ret, 12),
        "volume_ratio_12w": volume / volume.rolling(12).mean().replace(0, np.nan),
    }
    assert list(feats) == STATISTICAL_NAMES
    return feats
