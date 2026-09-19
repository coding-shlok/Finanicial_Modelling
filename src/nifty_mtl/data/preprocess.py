"""Daily -> weekly panel with quality checks and prediction targets.

Conventions
-----------
* Weeks are labelled by their Friday (``W-FRI``). If Friday is a holiday the
  week's last session is used as the close — no look-ahead.
* ``ret[t]``  = log(close[t] / close[t-1])   (weekly log return *of* week t)
* ``rv[t]``   = std(daily log returns inside week t) * sqrt(5)  (realized weekly vol)
* Targets for a decision made at the close of week t are ``ret[t+1]`` and
  ``rv[t+1]`` — see ``targets()``.
* ``valid[t, s]`` is True where symbol ``s`` has a clean bar in week t and
  passes the liquidity/price screens over the trailing year.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nifty_mtl.config import DATA_PROCESSED, DataConfig, Universe, load_universe
from nifty_mtl.data.fetch import fetch_daily

log = logging.getLogger(__name__)


@dataclass
class WeeklyPanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    ret: pd.DataFrame          # weekly log return
    rv: pd.DataFrame           # realized weekly vol from daily returns
    valid: pd.DataFrame        # bool mask
    bench_close: pd.Series
    bench_ret: pd.Series
    daily_ret: pd.DataFrame    # daily log returns (date x symbol), kept for graph features
    bench_daily_ret: pd.Series # daily benchmark log returns

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def weeks(self) -> pd.DatetimeIndex:
        return self.close.index

    def targets(self, clip: tuple[float, float] = (0.01, 0.10)) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Next-week return and volatility aligned to decision week t."""
        y_ret = self.ret.shift(-1)
        y_vol = self.rv.shift(-1).clip(*clip)
        return y_ret, y_vol

    def save(self, path=None) -> None:
        path = path or DATA_PROCESSED / "weekly.h5"
        with pd.HDFStore(path, mode="w", complevel=5) as st:
            for k in ["open", "high", "low", "close", "volume", "ret", "rv", "daily_ret"]:
                st.put(k, getattr(self, k))
            st.put("valid", self.valid.astype(int))
            st.put("bench_close", self.bench_close)
            st.put("bench_ret", self.bench_ret)
            st.put("bench_daily_ret", self.bench_daily_ret)
        log.info("saved weekly panel to %s", path)

    @classmethod
    def load(cls, path=None) -> "WeeklyPanel":
        path = path or DATA_PROCESSED / "weekly.h5"
        with pd.HDFStore(path, mode="r") as st:
            kw = {k: st[k] for k in ["open", "high", "low", "close", "volume", "ret", "rv", "daily_ret"]}
            kw["valid"] = st["valid"].astype(bool)
            kw["bench_close"] = st["bench_close"]
            kw["bench_ret"] = st["bench_ret"]
            kw["bench_daily_ret"] = st["bench_daily_ret"]
        return cls(**kw)


def _wide(daily: pd.DataFrame, field: str) -> pd.DataFrame:
    return daily[field].unstack("symbol")


def build_weekly(daily: pd.DataFrame | None = None, universe: Universe | None = None,
                 cfg: DataConfig | None = None) -> WeeklyPanel:
    universe = universe or load_universe()
    cfg = cfg or DataConfig()
    daily = daily if daily is not None else fetch_daily(universe)

    stocks = daily[daily.index.get_level_values("symbol") != "BENCH"]
    bench = daily.xs("BENCH", level="symbol")

    o, h, l, c, v = (_wide(stocks, f) for f in ["Open", "High", "Low", "Close", "Volume"])
    # keep the universe ordering
    cols = [s for s in universe.symbols if s in c.columns]
    o, h, l, c, v = (x[cols] for x in (o, h, l, c, v))

    # ---- forward fill short holiday/gap runs *within* the listed period only
    listed = c.notna().cumsum() > 0             # True after first observation
    c_ff = c.ffill().where(listed)
    o_ff, h_ff, l_ff = (x.ffill().where(listed) for x in (o, h, l))
    v_ff = v.fillna(0).where(listed)

    daily_ret = np.log(c_ff / c_ff.shift(1))

    # ---- weekly resample (W-FRI); label = Friday of that week
    rule = "W-FRI"
    wo = o_ff.resample(rule).first()
    wh = h_ff.resample(rule).max()
    wl = l_ff.resample(rule).min()
    wc = c_ff.resample(rule).last()
    wv = v_ff.resample(rule).sum()
    wret = np.log(wc / wc.shift(1))
    wrv = daily_ret.resample(rule).std(ddof=0) * np.sqrt(5)
    n_days = c.notna().resample(rule).sum()

    # ---- quality screens (PRD §5)
    price_ok = wc > cfg.min_price_inr
    avg_vol_52 = v_ff.rolling(252, min_periods=60).mean().resample(rule).last()
    liq_ok = avg_vol_52 > cfg.min_avg_daily_volume
    # missing-data rule: fraction of missing sessions over trailing 52 weeks
    miss_frac = 1 - (c.notna().rolling(252, min_periods=60).mean()).resample(rule).last()
    miss_ok = miss_frac < cfg.max_missing_frac
    has_bar = n_days > 0
    valid = (price_ok & liq_ok & miss_ok & has_bar & wc.notna() & wret.notna()).fillna(False)

    # drop leading rows where nothing is valid (index spin-up)
    first = valid.any(axis=1).idxmax()
    sl = slice(first, None)

    bench_close = bench["Close"].resample(rule).last()
    bench_ret = np.log(bench_close / bench_close.shift(1))

    panel = WeeklyPanel(
        open=wo.loc[sl], high=wh.loc[sl], low=wl.loc[sl], close=wc.loc[sl], volume=wv.loc[sl],
        ret=wret.loc[sl], rv=wrv.loc[sl], valid=valid.loc[sl],
        bench_close=bench_close.loc[sl], bench_ret=bench_ret.loc[sl],
        daily_ret=daily_ret,
        bench_daily_ret=np.log(bench["Close"]).diff(),
    )
    dropped = [s for s in cols if valid[s].mean() < 0.5]
    log.info("weekly panel: %d weeks x %d symbols; %d symbols valid <50%% of weeks: %s",
             len(panel.weeks), len(cols), len(dropped), dropped)
    return panel


def quality_report(panel: WeeklyPanel) -> pd.DataFrame:
    rows = []
    for s in panel.symbols:
        v = panel.valid[s]
        rows.append({
            "symbol": s,
            "first_valid": v.idxmax().date() if v.any() else None,
            "valid_weeks": int(v.sum()),
            "valid_frac": round(float(v.mean()), 3),
            "median_weekly_vol_inr": float((panel.volume[s] * panel.close[s]).median()),
            "nan_close": int(panel.close[s].isna().sum()),
        })
    return pd.DataFrame(rows).set_index("symbol")
