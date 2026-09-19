"""Failure analysis (PRD §9): where does the signal work and where does it break?

Slices the out-of-sample weekly rank-IC and long-only portfolio return by
market regime (volatility, correlation, trend) and the per-stock hit rate by
sector, beta bucket and size proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_mtl.backtest.engine import BacktestResult
from nifty_mtl.backtest.metrics import sharpe
from nifty_mtl.config import Universe
from nifty_mtl.features.build import FeatureSet


def _static_series(fs: FeatureSet, name: str, index: pd.DatetimeIndex) -> pd.Series:
    j = fs.static_names.index(name)
    t = fs.weeks.get_indexer(index)
    # market-wide features are identical across stocks: take the first usable stock
    vals = [fs.static[ti][fs.sample_ok[ti]][0, j] if fs.sample_ok[ti].any() else np.nan for ti in t]
    return pd.Series(vals, index=index)


def regime_table(ic: pd.Series, res: BacktestResult, fs: FeatureSet, rf_weekly: float,
                 bench_ret: pd.Series) -> pd.DataFrame:
    idx = ic.index
    regimes = {
        "market vol": _static_series(fs, "mkt_vol_20d", idx),
        "avg correlation": _static_series(fs, "mkt_avg_corr", idx),
        "market 12w trend": _static_series(fs, "mkt_ret_12w", idx),
        "dispersion": _static_series(fs, "mkt_dispersion", idx),
    }
    rows = []
    for name, s in regimes.items():
        hi = s > s.median()
        for label, m in [("high", hi), ("low", ~hi)]:
            r = res.ret.reindex(idx)[m]
            rows.append({"regime": name, "bucket": label, "weeks": int(m.sum()), "mean_ic": float(ic[m].mean()),
                         "ic_t_stat": float(ic[m].mean() / (ic[m].std() / np.sqrt(m.sum()))),
                         "port_sharpe": sharpe(r, rf_weekly), "port_mean_weekly_ret": float(r.mean()),
                         "bench_mean_weekly_ret": float(bench_ret.reindex(idx)[m].mean())})
    # also: weeks where the benchmark fell > 2%
    crash = bench_ret.reindex(idx) < -0.02
    for label, m in [("bench < -2%", crash), ("bench >= -2%", ~crash)]:
        r = res.ret.reindex(idx)[m]
        rows.append({"regime": "market week", "bucket": label, "weeks": int(m.sum()), "mean_ic": float(ic[m].mean()),
                     "ic_t_stat": float(ic[m].mean() / (ic[m].std() / np.sqrt(max(m.sum(), 2)))),
                     "port_sharpe": sharpe(r, rf_weekly), "port_mean_weekly_ret": float(r.mean()),
                     "bench_mean_weekly_ret": float(bench_ret.reindex(idx)[m].mean())})
    return pd.DataFrame(rows).set_index(["regime", "bucket"])


def stock_characteristic_table(score: pd.DataFrame, rr: pd.DataFrame, fs: FeatureSet, universe: Universe,
                               weekly_turnover_inr: pd.DataFrame) -> pd.DataFrame:
    """Per-stock precision of BUY signals and score-return correlation, grouped by
    sector, beta tercile and size (INR turnover) tercile."""
    idx = score.index
    rows = []
    j_beta = fs.static_names.index("beta_mkt")
    t_idx = fs.weeks.get_indexer(idx)
    for j, s in enumerate(fs.symbols):
        sc, r = score[s].reindex(idx), rr[s].reindex(idx)
        m = sc.notna() & r.notna()
        if m.sum() < 20:
            continue
        ranks = score.rank(axis=1, ascending=False)[s].reindex(idx)
        n_stocks = score.notna().sum(1).reindex(idx)
        buy = (ranks <= np.round(0.1 * n_stocks)) & m
        rows.append({
            "symbol": s, "sector": universe.sector_of[s],
            "beta": float(np.nanmean(fs.static[t_idx, j, j_beta])),
            "size_inr": float(weekly_turnover_inr[s].reindex(idx).median()),
            "weeks": int(m.sum()), "buy_signals": int(buy.sum()),
            "buy_hit_rate": float((r[buy] > 0).mean()) if buy.sum() else np.nan,
            "buy_mean_ret": float(r[buy].mean()) if buy.sum() else np.nan,
            "score_ret_corr": float(sc[m].corr(r[m], method="spearman")),
        })
    df = pd.DataFrame(rows).set_index("symbol")
    df["beta_bucket"] = pd.qcut(df["beta"], 3, labels=["low beta", "mid beta", "high beta"])
    df["size_bucket"] = pd.qcut(df["size_inr"], 3, labels=["smaller", "mid", "larger"])
    return df


def grouped_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=True)
    return pd.DataFrame({
        "n_stocks": g.size(),
        "buy_signals": g["buy_signals"].sum(),
        "buy_hit_rate": g.apply(lambda d: np.average(d["buy_hit_rate"].fillna(0), weights=np.maximum(d["buy_signals"], 1e-9))),
        "buy_mean_ret": g.apply(lambda d: np.average(d["buy_mean_ret"].fillna(0), weights=np.maximum(d["buy_signals"], 1e-9))),
        "mean_score_ret_corr": g["score_ret_corr"].mean(),
    })
