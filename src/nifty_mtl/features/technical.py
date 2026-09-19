"""Technical indicators on weekly bars (15 features, PRD §7).

All functions take wide DataFrames (index = week, columns = symbol) and are
strictly causal: the value at week t uses bars <= t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(x: pd.DataFrame, span: int) -> pd.DataFrame:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.DataFrame, fast=12, slow=26, sig=9):
    line = _ema(close, fast) - _ema(close, slow)
    signal = _ema(line, sig)
    hist = line - signal
    # scale by price so it is comparable across stocks
    return line / close, signal / close, hist / close


def bollinger(close: pd.DataFrame, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    width = (upper - lower) / ma
    return pct_b, width


def true_range(high, low, close):
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()]).groupby(level=0).max()


def atr(high, low, close, n: int = 14) -> pd.DataFrame:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / close


def obv(close, volume) -> pd.DataFrame:
    sign = np.sign(close.diff()).fillna(0)
    ob = (sign * volume).cumsum()
    # normalise: 12-week change in OBV relative to 12-week volume
    return (ob - ob.shift(12)) / volume.rolling(12).sum().replace(0, np.nan)


def cci(high, low, close, n: int = 20) -> pd.DataFrame:
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def stochastic(high, low, close, n: int = 14, d: int = 3):
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def williams_r(high, low, close, n: int = 14) -> pd.DataFrame:
    ll = low.rolling(n).min()
    hh = high.rolling(n).max()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def adx(high, low, close, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx_ = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx_, pdi, mdi


TECHNICAL_NAMES = [
    "rsi14", "macd", "macd_signal", "macd_hist", "bb_pctb", "bb_width", "atr14",
    "obv12", "cci20", "stoch_k", "stoch_d", "williams_r", "adx14", "plus_di", "minus_di",
]


def technical_features(open_, high, low, close, volume) -> dict[str, pd.DataFrame]:
    m_line, m_sig, m_hist = macd(close)
    bb_b, bb_w = bollinger(close)
    st_k, st_d = stochastic(high, low, close)
    adx_, pdi, mdi = adx(high, low, close)
    feats = {
        "rsi14": rsi(close),
        "macd": m_line, "macd_signal": m_sig, "macd_hist": m_hist,
        "bb_pctb": bb_b, "bb_width": bb_w,
        "atr14": atr(high, low, close),
        "obv12": obv(close, volume),
        "cci20": cci(high, low, close),
        "stoch_k": st_k, "stoch_d": st_d,
        "williams_r": williams_r(high, low, close),
        "adx14": adx_, "plus_di": pdi, "minus_di": mdi,
    }
    assert list(feats) == TECHNICAL_NAMES
    return feats
