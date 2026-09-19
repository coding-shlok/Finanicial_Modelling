"""Download daily OHLCV for the universe + benchmark and store as HDF5.

Daily data is fetched (not weekly) because the liquidity screen and the
realized-volatility target both need intra-week observations. Weekly bars are
built in ``preprocess.py``.
"""
from __future__ import annotations

import logging
import warnings

import pandas as pd
import yfinance as yf

from nifty_mtl.config import DATA_RAW, Universe, load_universe, Splits

log = logging.getLogger(__name__)
FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _apply_spinoff(df: pd.DataFrame, ex_date: str) -> pd.DataFrame:
    """Scale prices strictly before ``ex_date`` so a demerger is not a return.

    Factor = open on ex-date / close on the last session before it. This is the
    standard back-adjustment convention used by most vendors for spin-offs.
    """
    ex = pd.Timestamp(ex_date)
    before = df.index < ex
    if not before.any() or not (~before).any():
        return df
    prev_close = df.loc[before, "Close"].iloc[-1]
    ex_open = df.loc[~before, "Open"].iloc[0]
    factor = float(ex_open / prev_close)
    out = df.astype(float).copy()
    for c in ["Open", "High", "Low", "Close"]:
        out.loc[before, c] = out.loc[before, c] * factor
    # keep share-count consistent so INR turnover is comparable
    out.loc[before, "Volume"] = out.loc[before, "Volume"] / factor
    log.info("spin-off adjustment applied on %s: factor=%.4f", ex_date, factor)
    return out


def fetch_daily(universe: Universe | None = None, start: str | None = None,
                end: str | None = None, force: bool = False) -> pd.DataFrame:
    """Return a long-format DataFrame indexed by (date, symbol) with OHLCV.

    Cached to ``data/raw/daily.h5``.
    """
    universe = universe or load_universe()
    start = start or Splits().data_start
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache = DATA_RAW / "daily.h5"
    if cache.exists() and not force:
        return pd.read_hdf(cache, "daily")

    tick_to_sym = {v: k for k, v in universe.tickers.items()}
    tickers = list(universe.tickers.values()) + [universe.benchmark]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, interval="1d",
                          auto_adjust=True, progress=False, group_by="ticker", threads=True)

    frames = []
    for t in tickers:
        sym = tick_to_sym.get(t, "BENCH" if t == universe.benchmark else t)
        try:
            d = raw[t][FIELDS].dropna(subset=["Close"]).astype(float).copy()
        except KeyError:
            log.warning("no data for %s", t)
            continue
        if d.empty:
            log.warning("empty data for %s", t)
            continue
        if sym in universe.spinoffs:
            d = _apply_spinoff(d, universe.spinoffs[sym]["ex_date"])
        d["symbol"] = sym
        frames.append(d)

    daily = pd.concat(frames)
    daily.index.name = "date"
    daily = daily.reset_index().set_index(["date", "symbol"]).sort_index()
    daily.to_hdf(cache, key="daily", mode="w", complevel=5)
    log.info("saved %s rows for %d symbols to %s", len(daily), daily.index.get_level_values(1).nunique(), cache)
    return daily
